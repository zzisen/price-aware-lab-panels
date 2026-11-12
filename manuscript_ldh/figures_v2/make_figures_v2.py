"""Build the P2 submission figures from the supplied CSV files."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PANELS = OUT / "panels"
BRIEF = "figure specification"
sys.dont_write_bytecode = True
PLOT_CACHE = TemporaryDirectory(prefix=".plot-", dir=HERE)
os.environ["MPLCONFIGDIR"] = PLOT_CACHE.name

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.path import Path as DrawingPath
from matplotlib.textpath import TextToPath
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

MM = 1 / 25.4
PAD = 0.02
ROUTINE = "#8A8A85"
MATCHED = "#2A78D6"
AWARE = "#D9822B"
SCORE = "#A0785A"
INK = "#1D1D1B"
INK2 = "#55554F"
LIGHT = "#E5E3DC"
WARM = "#C0392B"
WHITE = "white"
DASH = (0, (3.2, 2.5))
SERIES = {"Routine bundle": ROUTINE, "Matched panel": MATCHED, "Target-aware panel": AWARE}
COHORTS = ("NHANES III", "NHANES 2011–14", "CHNS", "Understanding Society", "CLHLS (65–79 y)")
NHANES = ("NHANES III", "NHANES 2011–14", "NHANES 2015–16")
mpl.rcParams.update({
    "font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": False, "axes.facecolor": WHITE, "figure.facecolor": WHITE,
    "savefig.facecolor": WHITE, "axes.unicode_minus": True,
})

# Years and other ranges have an en dash or hyphen as a separator. A numerical
# minus uses U+2212. Preserve separators within a printed thousands group.
NUMBERS = re.compile(r"[+−]?\d+(?:[,\u202f ]\d{3})*(?:\.\d+)?(?:%|×)?|²")
FONT_METRICS = TextToPath()


@dataclass(frozen=True)
class Source:
    file: str
    row: str
    column: str


@dataclass(frozen=True)
class Label:
    text: str
    source: Source


class Table:
    def __init__(self, filename):
        self.filename = filename
        self.rows = pd.read_csv(HERE / "data" / filename, encoding="utf-8-sig")

    def source(self, rows, column):
        return Source(self.filename, "|".join(str(int(i)) for i in rows), column)

    def label(self, row, column, formatter=str):
        return Label(formatter(row[column]), self.source([row.name], column))

    def derived(self, rows, column, text):
        return Label(text, self.source(rows, column))


def find(rows, **values):
    result = rows
    for column, value in values.items():
        result = result[result[column].eq(value)]
    if len(result) != 1:
        raise ValueError(f"Expected one row for {values}, found {len(result)}")
    return result.iloc[0]


def label_parts(label):
    return (label,) if isinstance(label, (str, Label)) else label


def plain(label):
    return "".join(part.text if isinstance(part, Label) else part for part in label_parts(label))


def width_mm(value, size, bold=False):
    font = FontProperties(family="Arial", size=size, weight="bold" if bold else "normal")
    width, _, _ = FONT_METRICS.get_text_width_height_descent(value, font, ismath=False)
    return width * 25.4 / 72


def wrap(value, max_mm, size, bold=False):
    result = []
    for line in value.split("\n"):
        line = re.sub(r"(?<=\d) (?=\d{3}(?:\D|$))", "\u00a0", line)
        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}".strip()
            if current and width_mm(candidate, size, bold) > max_mm:
                result.append(current.replace("\u00a0", " "))
                current = word
            else:
                current = candidate
        result.append(current.replace("\u00a0", " "))
    return "\n".join(result)


def decimal(value, places=3):
    if not value:
        return "0"
    return f"{value:.{places}f}".rstrip("0").rstrip(".").replace("-", "−")


def percent(value):
    return f"{float(value):.0%}"


def usd(value):
    return f"US${float(value):.2f}"


def tint(color, fraction):
    rgb = to_rgb(color)
    return tuple(1 - fraction * (1 - channel) for channel in rgb)


def nice_ticks(lower, upper, bins):
    ticks = MaxNLocator(nbins=bins, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper)
    return [float(tick) for tick in ticks if lower - 1e-9 <= tick <= upper + 1e-9]


def max_cell(rows, columns):
    values = rows[columns].to_numpy(dtype=float)
    ri, ci = np.unravel_index(np.nanargmax(values), values.shape)
    return float(values[ri, ci]), rows.index[ri], columns[ci]


class Audit:
    fields = ("figure", "panel", "printed_text", "source_csv", "source_row", "source_column")

    def __init__(self):
        self.rows = []

    def text(self, figure, panel, label, context):
        pieces = label_parts(label)
        rendered = plain(pieces)
        origins = []
        for part in pieces:
            if isinstance(part, Label):
                origins.extend([part.source] * len(part.text))
            else:
                origins.extend([Source(BRIEF, f"{figure} {panel}".strip(), context)] * len(part))
        for token in NUMBERS.finditer(rendered):
            # Each numeric value is one label part. A currency prefix or a
            # fixed sign may be in an adjacent part.
            source = origins[next(i for i in range(*token.span()) if rendered[i].isdigit())]
            self.rows.append((figure, panel, token.group(), source.file, source.row, source.column))
        return rendered

    def check_text(self, figure, labels):
        expected = Counter(row[2] for row in self.rows if row[0] == figure)
        actual = Counter(token for label in labels for token in NUMBERS.findall(label))
        if expected != actual:
            raise ValueError(f"{figure} numeric text differs from ledger: "
                             f"unlisted {dict(actual - expected)}, missing {dict(expected - actual)}")

    def save(self):
        with (OUT / "figure_numbers_used.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(self.fields)
            writer.writerows(self.rows)


@dataclass(frozen=True)
class Region:
    left: float
    top: float
    width: float
    height: float


class Plate:
    def __init__(self, name, width, height, audit):
        self.name = name
        self.width = width
        self.height = height
        self.audit = audit
        self.figure = plt.figure(figsize=(width * MM, height * MM), dpi=300)
        self.panels = []

    def axes(self, region):
        r = region
        position = [r.left / self.width, 1 - (r.top + r.height) / self.height,
                    r.width / self.width, r.height / self.height]
        return self.figure.add_axes(position, facecolor="none")

    def panel(self, key, region, title=None, crop=None):
        pane = Panel(self, key, region, crop or region)
        self.panels.append(pane)
        if title:
            pane.heading(title)
        return pane

    def _write(self, stem, formats, region):
        # Tight export, with an invisible inner boundary so padding restores
        # the exact physical extent even when the content has spare margins.
        edge = PAD / MM
        r = region
        guide = Rectangle(((r.left + edge) / self.width,
                           1 - (r.top + r.height - edge) / self.height),
                          (r.width - 2 * edge) / self.width,
                          (r.height - 2 * edge) / self.height,
                          transform=self.figure.transFigure, facecolor="none",
                          edgecolor="none", linewidth=0)
        self.figure.add_artist(guide)
        try:
            for extension in formats:
                self.figure.savefig(stem.with_suffix(f".{extension}"),
                                    bbox_inches="tight", pad_inches=PAD, dpi=300)
        finally:
            guide.remove()

    def save(self):
        self._write(OUT / self.name, ("pdf", "svg", "png"), Region(0, 0, self.width, self.height))
        root = ET.parse(OUT / f"{self.name}.svg").getroot()
        svg_text = ["".join(node.itertext()) for node in root.iter("{http://www.w3.org/2000/svg}text")]
        if not svg_text:
            raise ValueError(f"{self.name} SVG has no editable text")
        self.audit.check_text(self.name, svg_text)

        for pane in self.panels:
            if not pane.key:
                continue
            for other in self.panels:
                for axes in other.artists:
                    axes.set_visible(other is pane)
            self._write(PANELS / f"{self.name}_{pane.key}", ("pdf", "svg"), pane.crop)
        plt.close(self.figure)


class Panel:
    def __init__(self, plate, key, region, crop):
        self.plate = plate
        self.key = key
        self.region = region
        self.crop = crop
        # Inset the axis bounds by the export padding while retaining
        # millimetre coordinates relative to the nominal panel region.
        inset = PAD / MM
        self.base = plate.axes(Region(region.left + inset, region.top + inset,
                                      region.width - 2 * inset, region.height - 2 * inset))
        self.base.set(xlim=(inset, region.width - inset), ylim=(region.height - inset, inset))
        self.base.set_axis_off()
        self.artists = [self.base]

    def plot(self, x, y, w, h):
        r = self.region
        ax = self.plate.axes(Region(r.left + x, r.top + y, w, h))
        self.artists.append(ax)
        return ax

    def note(self, label, context="fixed text"):
        return self.plate.audit.text(self.plate.name, self.key or "whole", label, context)

    def text(self, x, y, label, *, ax=None, size=6.5, bold=False,
             max_width=None, ha="left", va="center", color=INK, **kwargs):
        value = self.note(label)
        if max_width is not None:
            value = wrap(value, max_width, size, bold)
        return (self.base if ax is None else ax).text(
            x, y, value, fontsize=size, fontweight="bold" if bold else "normal",
            ha=ha, va=va, color=color, clip_on=False, linespacing=1.10, **kwargs)

    def heading(self, title):
        start = 1.8
        if self.key:
            self.text(start, 3.35, self.key, size=9, bold=True)
            start = 6.6
        self.text(start, 3.35, title, size=8, bold=True)

    def ticks(self, ax, direction, values, labels=None):
        labels = labels if labels is not None else [decimal(value) for value in values]
        rendered = [self.note(label, "axis tick") for label in labels]
        if direction == "x":
            ax.set_xticks(values, rendered)
        else:
            ax.set_yticks(values, rendered)

    def axis_title(self, ax, direction, label, **kwargs):
        setter = ax.set_xlabel if direction == "x" else ax.set_ylabel
        setter(self.note(label, "axis title"), **kwargs)

    def legend(self, handles, labels, x, y, *, center=False, **kwargs):
        labels = [self.note(label, "legend") for label in labels]
        return self.base.legend(handles, labels, bbox_to_anchor=(x, y),
                                bbox_transform=self.base.transData, ncols=len(labels),
                                loc="upper center" if center else "upper left",
                                borderaxespad=0, borderpad=0, handletextpad=0.45,
                                columnspacing=0.9, handlelength=1.3, **kwargs)


def box(ax, x, y, width, height, fill=WHITE, edge=INK):
    artist = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.4,rounding_size=1.5",
                            linewidth=0.6, facecolor=fill, edgecolor=edge)
    ax.add_patch(artist)
    return artist


def arrow(ax, start, end, *, color=INK2, linewidth=0.8, scale=6,
          shrinkA=0, shrinkB=0, path=None, zorder=4):
    style = dict(arrowstyle="-|>", color=color, linewidth=linewidth,
                 mutation_scale=scale, shrinkA=shrinkA, shrinkB=shrinkB,
                 clip_on=False, zorder=zorder)
    patch = (FancyArrowPatch(path=DrawingPath(path), **style) if path else
             FancyArrowPatch(start, end, **style))
    ax.add_patch(patch)


def error_point(ax, x, y, lo, hi, color, *, horizontal=True, hollow=False, size=4):
    coords = dict(xerr=[[x - lo], [hi - x]]) if horizontal else dict(yerr=[[y - lo], [hi - y]])
    ax.errorbar([x], [y], fmt="o", markersize=size,
                markerfacecolor=WHITE if hollow else color,
                markeredgecolor=color, markeredgewidth=0.65, color=color,
                ecolor=color, elinewidth=0.9, capsize=0, linestyle="none",
                zorder=4, **coords)


def forest_estimate(ax, row, y, color, kind):
    if kind == "pooled":
        ax.add_patch(Polygon([(row.lo, y), (row.est, y - 0.24), (row.hi, y),
                              (row.est, y + 0.24)], closed=True,
                             facecolor=INK, edgecolor="none", zorder=5))
    else:
        error_point(ax, row.est, y, row.lo, row.hi,
                    ROUTINE if kind == "supplementary" else color,
                    hollow=kind == "supplementary")


def forest_axes(ax, ylim, xlim, margin=None, *, zero=True):
    ax.set(xlim=xlim, ylim=ylim)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    if zero:
        ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    if margin is not None:
        ax.axvline(margin, color=INK2, linestyle=DASH, linewidth=0.6, zorder=1)


def panel_plot_style(ax):
    ax.tick_params(axis="both", pad=2)



def lancet_n(n):
    return f"{n:,}".replace(",", " ") if n >= 10000 else str(n)


def figure1(audit):
    plate = Plate("Figure1", 180, 125, audit)
    a = plate.panel("A", Region(0, 0, 55, 55), "CMP fee: panel or single tests")
    b = plate.panel("B", Region(55, 0, 125, 55), "Panels chosen in training data")
    c = plate.panel("C", Region(0, 55, 180, 50), "Training and validation cohorts")
    d = plate.panel("D", Region(0, 105, 180, 20), "Prespecified hypotheses")
    prices = Table("F1A_cmp_price.csv")
    cards = Table("F1C_cohort_cards.csv")
    train = find(cards.rows, order=0)
    top, bottom = (prices.rows.iloc[i] for i in range(2))
    n_items = prices.label(top, "item", lambda s: re.search(r"\d+", s).group())
    for row, y, name, color in [(top, 15, "CMP billed as one panel", MATCHED),
                                 (bottom, 30, ("Its ", n_items, " tests billed singly"), ROUTINE)]:
        length = row.usd / bottom.usd * 31.0
        a.text(2.5, y - 2.3, name)
        a.base.add_patch(Rectangle((2.9, y + 1.2), length, 3.6, facecolor=color, edgecolor="none"))
        a.text(2.9 + length + 1.6, y + 3.0, prices.label(row, "usd", usd))
    a.text(2.5, 43.0, (prices.derived([bottom.name, top.name], "usd: first listed row ÷ second, 1 decimal",
                                      f"{bottom.usd / top.usd:.1f}×"), " the panel fee"), size=8, bold=True)
    box(b.base, 2.6, 15.4, 32, 23.2, fill=LIGHT)
    box(b.base, 40.3, 15.4, 32, 23.2)
    b.text(4.2, 19.5, cards.label(train, "cohort"), bold=True)
    b.text(4.2, 24.4, (cards.label(train, "n", lancet_n), " adults aged 40–79"), max_width=29.2)
    b.text(4.2, 29.3, (cards.label(train, "deaths"), " deaths in ", cards.label(train, "follow_up")))
    b.text(4.2, 34.3, (cards.label(train, "menu_analytes"), " candidate analytes"))
    b.text(41.9, 19.5, "Price-aware search", bold=True)
    for y, title in [(24.4, "Ridge Cox model"), (29.3, "Adds a panel or a single test"),
                     (34.3, "Priced by the fee schedule")]:
        b.text(41.9, y, title, max_width=29.0)
    arrow(b.base, (35.1, 27.0), (39.8, 27.0))

    outputs = Table("F1B_outputs.csv")
    matched = find(outputs.rows, output="Matched panel (USA)")
    aware = find(outputs.rows, output="Target-aware panel (USA)")
    chart = find(outputs.rows, output="Ordering chart")
    out_notes = [
        ("Matched panel", MATCHED, tint(MATCHED, 0.12), (
            outputs.label(matched, "items"), " · ", outputs.label(matched, "usd", usd),
            " (", outputs.label(matched, "share_of_routine", percent), " of routine)")),
        ("Target-aware panel", AWARE, tint(AWARE, 0.12), (
            outputs.label(aware, "items"), " · ", outputs.label(aware, "usd", usd), " (within routine budget)")),
        ("Ordering chart", INK2, LIGHT, outputs.label(
            chart, "items", lambda text: text.replace("clinical information", "Clinical data")
            .replace(" -> ", " → ").replace(" → urine ACR", "\n→ urine ACR"))),
    ]
    for y, (title, color, fill, desc) in zip((8.7, 24.0, 39.3), out_notes, strict=True):
        box(b.base, 77.2, y, 45, 13, fill, color)
        b.text(78.8, y + 2.85, title, bold=True)
        b.text(78.8, y + 8.4, desc, max_width=41.8)
        arrow(b.base, (72.8, 27.0), (76.7, y + 6.5))

    price_labels = {"US Medicare 2026": "US", "Shandong public prices 2026": "Chinese",
                    "US Medicare 2026 (scenario)": "US (scenario)"}
    deaths_labels = {"National Death Index linkage": "Death index",
                     "Death year or survey wave of first report": "Year/wave reported",
                     "Verified death dates": "Verified dates",
                     "Survey wave in which death was reported": "Wave reported"}
    role_styles = {"training": (INK, "Training"), "primary": (MATCHED, "Primary validation"),
                   "supplementary": (ROUTINE, "Supplementary")}
    attrs = [
        ("Country, years", lambda r: (cards.label(r, "country"), ", ", cards.label(r, "years"))),
        ("Participants", lambda r: cards.label(r, "n", lancet_n)),
        ("Deaths", lambda r: cards.label(r, "deaths", lancet_n)),
        ("Analytes", lambda r: cards.label(r, "menu_analytes")),
        ("Prices", lambda r: cards.label(r, "prices", price_labels.__getitem__)),
        ("Deaths recorded by", lambda r: cards.label(r, "death_ascertainment", deaths_labels.__getitem__)),
        ("Follow-up", lambda r: cards.label(r, "follow_up")),
    ]
    rows = cards.rows.sort_values("order")
    x0, lw, right = 2.1, 21.0, 177.9
    cw = (right - x0 - lw) / len(rows)
    band_y, top, row_h = 14.6, 18.6, 4.2
    for j, (name, _) in enumerate(attrs):
        y = top + row_h * (j + 0.5)
        if j % 2 == 0:
            c.base.add_patch(Rectangle((x0, y - row_h / 2), right - x0, row_h, facecolor=tint(LIGHT, 0.55),
                                       edgecolor="none", zorder=0))
        c.text(x0 + 0.8, y, name, size=6, color=INK2)
    for i, (_, r) in enumerate(rows.iterrows()):
        left = x0 + lw + i * cw
        cx = left + cw / 2
        c.text(cx, 9.6, cards.label(r, "cohort"), size=6.3, bold=True, ha="center", max_width=cw - 1.0)
        color, text = role_styles[r.role]
        c.base.add_patch(Rectangle((left + 0.25, band_y), cw - 0.5, 3.0, facecolor=color, edgecolor="none"))
        c.text(cx, band_y + 1.5, text, size=5.5, color=WHITE, ha="center")
        for j, (_, cell) in enumerate(attrs):
            c.text(cx, top + row_h * (j + 0.5), cell(r), size=6, ha="center", max_width=cw - 0.8)

    questions = [
        "Cheaper panel, non-inferior discrimination (margin −0.01)",
        "Better discrimination within the routine budget",
        "Ordering in steps: cost ≤50% of routine",
        "Versus CoAI at equal budgets",
    ]
    qw = (180 - 4.2 - 3 * 2) / 4
    for i, question in enumerate(questions, 1):
        left = 2.1 + (i - 1) * (qw + 2)
        box(d.base, left, 7.9, qw, 10.3)
        d.text(left + 1.45, 13.05, f"H{i}", bold=True)
        d.text(left + 6.7, 13.05, question, max_width=qw - 8.1)
    plate.save()


SCATTER_LABELS = {
    "Age and sex only": ("Age and sex only", 7.0, -1.5, "left"),
    "BMP": ("BMP", 4.0, 0.0, "left"),
    "CBC": ("CBC", -4.0, 0.0, "right"),
    "CMP": ("CMP", -4.0, 1.5, "right"),
    "Lipids": ("Lipids", 4.0, 0.0, "left"),
    "CMP+CBC": ("CMP+CBC", -3.0, 5.5, "right"),
    "CBC+BMP (Intermountain-type)": ("CBC+BMP (Intermountain-type)", 4.0, -1.5, "left"),
    "PhenoAge analytes (refitted)": ("PhenoAge", 4.0, -1.0, "left"),
    "Routine bundle": ("Routine bundle", 5.0, 5.0, "left"),
    "Matched panel": ("Matched panel", 0.0, -9.0, "center"),
    "Target-aware panel": ("Target-aware panel", -6.0, 7.0, "right"),
    "All 34 analytes": ("All 34 analytes", 0.0, -8.0, "center"),
}


def scatter_frontier(pane, data, death_years):
    ax = pane.plot(15.0, 9.5, pane.region.width - 19.0, 52.0)
    panel_plot_style(ax)
    ax.set(xlim=(-2, 110), ylim=(0.738, 0.816))
    pane.ticks(ax, "x", [0, 20, 40, 60, 80, 100])
    yticks = [0.74, 0.76, 0.78, 0.80]
    pane.ticks(ax, "y", yticks, [f"{t:.2f}" for t in yticks])
    pane.axis_title(ax, "x", "Laboratory fee per person (US$)", labelpad=2.0)
    pane.axis_title(ax, "y", f"Harrell's C ({death_years}-year death)", labelpad=2.0)
    rows = data.rows.sort_values("usd")
    frontier, best = [], -1.0
    for _, row in rows.iterrows():
        if row.C > best:
            frontier.append(row); best = row.C
    ax.plot([r.usd for r in frontier], [r.C for r in frontier], color=tint(INK2, 0.45), linewidth=1.6,
            solid_capstyle="round", zorder=2)
    routine = find(data.rows, category="routine")
    matched = find(data.rows, category="matched")
    aware = find(data.rows, category="target-aware")
    ax.axhline(routine.C, color=ROUTINE, linewidth=0.5, linestyle=DASH, zorder=1)
    ax.axvline(routine.usd, color=ROUTINE, linewidth=0.5, linestyle=DASH, zorder=1)
    styles = {"standard bundle": (ROUTINE, "o", 4, WHITE), "reference line": (ROUTINE, "o", 4, WHITE),
              "routine": (ROUTINE, "o", 6, ROUTINE), "established score": (SCORE, "s", 5, SCORE),
              "matched": (MATCHED, "o", 8, MATCHED), "target-aware": (AWARE, "o", 8, AWARE)}
    for _, row in data.rows.iterrows():
        color, marker, size, fill = styles[row.category]
        ax.plot(row.usd, row.C, marker=marker, markersize=size, color=color,
                markerfacecolor=fill, markeredgewidth=0.7, linestyle="none", zorder=6)
        if row.category in ("matched", "target-aware", "routine"):
            continue                                   # identified in the legend box
        label, dx, dy, align = SCATTER_LABELS[row.panel]
        ax.annotate(pane.note(label), (row.usd, row.C), (dx, dy), textcoords="offset points",
                    ha=align, va="center", fontsize=6.2, color=INK2, annotation_clip=False, zorder=7)
    arrow(ax, (routine.usd, routine.C), (matched.usd, routine.C),
          color=INK, linewidth=0.7, shrinkA=3.2, shrinkB=4.2, scale=7)
    arrow(ax, (routine.usd, routine.C), (routine.usd, aware.C),
          color=INK, linewidth=0.7, shrinkA=3.2, shrinkB=4.2, scale=7)
    c3 = lambda v: f"{v:.3f}"
    entries = [
        (Line2D([], [], marker="o", markersize=8, color=MATCHED, linestyle="none"),
         ("Matched panel: ", data.label(matched, "usd", usd), ", C ", data.label(matched, "C", c3))),
        (Line2D([], [], marker="o", markersize=8, color=AWARE, linestyle="none"),
         ("Target-aware panel: ", data.label(aware, "usd", usd), ", C ", data.label(aware, "C", c3))),
        (Line2D([], [], marker="o", markersize=6, color=ROUTINE, linestyle="none"),
         ("Routine bundle: ", data.label(routine, "usd", usd), ", C ", data.label(routine, "C", c3))),
        (Line2D([], [], color=tint(INK2, 0.45), linewidth=1.6), "Panels not beaten by a cheaper panel"),
    ]
    ax.legend([h for h, _ in entries], [pane.note(t, "legend") for _, t in entries], loc="lower right",
              frameon=False, fontsize=6.8, handlelength=1.8, handletextpad=0.6, labelspacing=0.7,
              borderaxespad=0.6)


FOREST_LABELS = {
    "Pooled (random effects)": "Pooled (random effects)",
    "NHANES 2015–16 (supplementary)": "NHANES 2015–16\n(supplementary)",
}


def validation_forest(pane, data, hypothesis, matched):
    x0, width, fee_x = (27.5, 46.0, 82.5) if matched else (5.0, 68.5, 82.5)
    ax = pane.plot(x0, 15.5, width, 46.0)
    rows = data.rows[data.rows.hypothesis.eq(hypothesis)]
    ys = [0, 1, 2, 3, 4, 5, 6.35]
    color = MATCHED if matched else AWARE
    if matched:
        forest_axes(ax, (7.0, -0.7), (-0.0125, 0.0085), -0.01)
        pane.ticks(ax, "x", [-0.01, -0.005, 0, 0.005])
        pane.ticks(ax, "y", ys, [data.label(row, "cohort", lambda s: FOREST_LABELS.get(s, s)) for _, row in rows.iterrows()])
        pane.text(-0.01, 1.03, "Margin", ax=ax, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK2)
    else:
        forest_axes(ax, (7.0, -0.7), (-0.02, 0.045))
        pane.ticks(ax, "x", [-0.02, 0, 0.02, 0.04])
        ax.set_yticks([])
    pane.text(0, 1.03, " better →", ax=ax, transform=ax.get_xaxis_transform(), ha="left", va="bottom", color=INK2)
    pane.axis_title(ax, "x", "Difference in discrimination (C or AUC)", labelpad=2)
    ax.tick_params(axis="x", pad=2)
    ax.axhline(5.68, linewidth=0.5, color=LIGHT)
    share_col = "matched_price_share" if matched else "aware_price_share"
    frac = (fee_x - x0) / width
    pane.text(frac, -1.25, "Fee, % of\nroutine", ax=ax, transform=ax.get_yaxis_transform(),
              ha="center", bold=True, color=color)
    for y, (_, row) in zip(ys, rows.iterrows(), strict=True):
        forest_estimate(ax, row, y, color, row.kind)
        if row.kind != "pooled":
            pane.text(frac, y, data.label(row, share_col, percent), ax=ax,
                      transform=ax.get_yaxis_transform(), ha="center", bold=True, color=color)
        elif not matched:
            pane.text(row.est, y + 0.5, ("I² = ", data.label(row, "I2", percent)),
                      ax=ax, ha="center", color=INK2)


def figure2(audit):
    plate = Plate("Figure2", 180, 150, audit)
    a = plate.panel("A", Region(0, 0, 180, 72), "Fee and discrimination of laboratory panels in NHANES III")
    b = plate.panel("B", Region(0, 74, 90, 76), "Matched: lower fee, non-inferior discrimination")
    c = plate.panel("C", Region(90, 74, 90, 76), "Target-aware: better within the routine budget")
    scatter_frontier(a, Table("F2A_cost_C_NHANES3.csv"), 9)
    forest = Table("F2BC_forest.csv")
    validation_forest(b, forest, "H1", True)
    validation_forest(c, forest, "H2", False)
    plate.save()



def cheapest_matrix(pane):
    matrix = Table("F3A_panel_items.csv")
    columns = [
        ("USA", "USA", 19.8), ("Australia", "Australia", 32.15), ("Japan", "Japan", 42.8),
        ("China, CHNS menu", "China\n(CHNS menu)", 52.2),
        ("China, CLHLS menu", "China\n(CLHLS menu)", 61.3),
        ("UK menu", "UK menu\n(US prices)", 72.35),
        ("Cardiovascular death (USA)", "CVD death\n(USA)", 86.1),
    ]
    items = [
        ("CMP", "CMP (panel)"), ("CBC", "CBC"), ("phosphorus", "Phosphorus"),
        ("ggt", "GGT"), ("ldh", "LDH"), ("hdl", "HDL cholesterol"),
        ("uric_acid", "Uric acid"), ("glucose", "Glucose"), ("bun", "Urea"),
        ("creatinine", "Creatinine"), ("albumin", "Albumin"), ("protein", "Total protein"),
    ]
    pitch = 5.58
    side = 0.8 * pitch
    for i, (item, item_label) in enumerate(items):
        present = matrix.rows[matrix.rows.item.eq(item)]
        if present.empty:
            continue
        y = 11.4 + i * pitch
        pane.text(17.05, y, item_label, ha="right")
        for column, _, x in columns:
            match = present[present.column.eq(column)]
            pane.base.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                                          facecolor=LIGHT if match.empty else MATCHED,
                                          edgecolor="none", clip_on=False))
            if not match.empty and match.iloc[0]["how"] == "6 measured CMP analytes":
                pane.text(x, y, (matrix.label(match.iloc[0], "how",
                          lambda val: val.split()[0]), "/14"), size=6, ha="center", color=WHITE)
    pane.base.plot([17.55, 88.4], [11.4 + 1.5 * pitch] * 2, color=LIGHT,
                   linewidth=0.5, clip_on=False)
    for column, heading, x in columns:
        row = matrix.rows[matrix.rows.column.eq(column)].iloc[0]
        title, _, qualifier = heading.partition("\n")
        pane.text(x, 77.45, title, size=6, ha="center")
        if qualifier:
            pane.text(x, 83.0 if column == "China, CLHLS menu" else 80.35,
                      qualifier, size=6, ha="center")
        pane.text(x, 87.75, matrix.label(row, "panel_price"), size=6, ha="center")
        reference = matrix.label(row, "reference_price",
                                 lambda text: text.replace(" (PREVENT full set)", ""))
        pane.text(x, 91.4, reference, size=6, color=INK2, ha="center")
    pane.text(2, 87.75, "Panel fee", size=6)
    pane.text(2, 91.4, "Reference", size=6, color=INK2)


def coai_forest(pane):
    data = Table("F3B_coai.csv")
    ax = pane.plot(29.5, 7.5, 53.3, 28.5)
    forest_axes(ax, (10.0, -0.4), (-0.005, 0.032))
    ax.tick_params(axis="x", pad=1.6)
    pane.ticks(ax, "x", [0, 0.01, 0.02, 0.03])
    pane.axis_title(ax, "x", "Difference in C or AUC", labelpad=1.2)
    locations = []
    labels = []
    for budget, y in [("matched-panel budget", 0.82), ("routine budget", 6.04)]:
        rows = data.rows[data.rows.budget.eq(budget)]
        pane.text(-0.52, y - 1.04, budget.capitalize(), ax=ax,
                  transform=ax.get_yaxis_transform(), bold=True)
        for i, (_, row) in enumerate(rows.iterrows()):
            location = y + i * 0.96
            locations.append(location)
            labels.append(data.label(row, "cohort"))
            forest_estimate(ax, row, location, INK, "pooled" if pd.isna(row.metric) else "cohort")
    pane.ticks(ax, "y", locations, labels)


def cardiovascular(pane):
    data = Table("F3C_cvd.csv")
    for i, name in enumerate(NHANES[:2]):
        rows = data.rows[data.rows.cohort.eq(name)]
        ax = pane.plot(39.0 + 23.3 * i, 15.0, 20.3, 21.3)
        forest_axes(ax, (4.45, -0.45), (0.76, 0.86), zero=False)
        ax.tick_params(axis="x", pad=1.8)
        pane.ticks(ax, "x", [0.78, 0.82, 0.86])
        pane.axis_title(ax, "x", "Harrell's C", labelpad=1.4)
        row = rows.iloc[0]
        pane.text(49.15 + 23.3 * i, 9.95,
                  (data.label(row, "cohort"), "\n(", data.label(row, "cvd_deaths"), " deaths)"),
                  size=7, ha="center")
        if i == 0:
            labels = [
                data.label(r, "panel") if r.usd == 0
                else (data.label(r, "panel"), " (", data.label(r, "usd", usd), ")")
                for _, r in rows.iterrows()
            ]
            pane.ticks(ax, "y", range(len(rows)), labels)
        else:
            ax.set_yticks([])
        for y, (_, row) in enumerate(rows.iterrows()):
            target = row.panel == "Target-aware panel"
            ax.plot(row.C, y, "o", markersize=6 if target else 4,
                    color=AWARE if target else ROUTINE, markeredgewidth=0)


def figure3(audit):
    plate = Plate("Figure3", 180, 100, audit)
    a = plate.panel("A", Region(0, 0, 88, 100), "Cheapest panel matching the reference",
                    crop=Region(0, 0, 93.3, 100))
    b = plate.panel("B", Region(94, 0, 86, 46), "Our panels minus CoAI")
    c = plate.panel("C", Region(94, 54, 86, 46), "Cardiovascular death")
    cheapest_matrix(a)
    coai_forest(b)
    cardiovascular(c)
    plate.save()


def order_chart(pane):
    chart = Table("F4A_chart.csv")
    ax = pane.plot(24.0, 15.0, 44.0, 40.6)
    panel_plot_style(ax)
    bounds = (2.65, -0.65)
    ax.set(xlim=(0, 0.25), ylim=bounds)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    pane.ticks(ax, "y", [0, 1, 2], ["Clinical data only", "After CMP", "After CMP and CBC"])
    ticks = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
    pane.ticks(ax, "x", ticks, [percent(v) for v in ticks])
    pane.axis_title(ax, "x", "Predicted 9-year risk of death", labelpad=2)
    ax.axvline(0.10, color=INK2, linewidth=0.6, linestyle=DASH, zorder=2)
    pane.text(0.10, 1.035, "Decision threshold", ax=ax, transform=ax.get_xaxis_transform(),
              ha="center", va="bottom", color=INK2)
    for y, ((_, row), share_name, color) in enumerate(zip(
            chart.rows.iterrows(), ("clinical only", "CMP", "CMP+CBC"),
            (MATCHED, tint(MATCHED, 0.70), tint(MATCHED, 0.45)), strict=True)):
        ax.barh(y, row.risk_high - row.risk_low, left=row.risk_low,
                height=4 * (bounds[0] - bounds[1]) / 40.6,
                color=color, edgecolor="none", zorder=1)
        pane.text(row.risk_high + 0.010, y,
                  ("→ order ", chart.label(row, "next_test"), " (", chart.label(row, "next_fee_usd", usd), ")"),
                  ax=ax, max_width=28, color=INK)


EARLIER = "Earlier rule"
VALUE = "Value-of-information rule"


def ordering_outcomes(pane, data, fee):
    ax = pane.plot(12.2 if fee else 10.6, 15.0, 31.2 if fee else 30.9, 40.6)
    panel_plot_style(ax)
    x = np.arange(len(NHANES))
    xlabels = [data.label(find(data.rows, cohort=name, rule=EARLIER), "cohort",
                          lambda text: text.replace("NHANES ", "NHANES\n", 1)) for name in NHANES]
    pane.ticks(ax, "x", x, xlabels)
    ax.set_xlim(-0.31, 2.31)
    ax.tick_params(axis="x", length=0, pad=2)
    for rule, color, shift in [(EARLIER, INK2, -0.15), (VALUE, MATCHED, 0.15)]:
        for i, name in enumerate(NHANES):
            row = find(data.rows, cohort=name, rule=rule)
            pos = i + shift
            if fee:
                value = row.cost_share
                ax.bar(pos, value, width=0.26, color=LIGHT if rule == EARLIER else MATCHED,
                       edgecolor=INK2 if rule == EARLIER else MATCHED, linewidth=0.6, zorder=3)
                ax.errorbar(pos, value, yerr=[[value - row.cost_share_lo], [row.cost_share_hi - value]],
                            fmt="none", ecolor=INK, elinewidth=0.9, capsize=0, zorder=5)
            else:
                error_point(ax, pos, row.extra_missed_per1000, row.extra_missed_lo,
                            row.extra_missed_hi, color, horizontal=False, hollow=rule == EARLIER)
    if fee:
        ax.set_ylim(0, 0.6)
        ticks = [0, 0.2, 0.4, 0.6]
        pane.ticks(ax, "y", ticks, [percent(v) for v in ticks])
        pane.axis_title(ax, "y", "Fee as % of routine bundle", labelpad=1.5)
        ax.axhline(0.50, linewidth=0.6, linestyle=DASH, color=INK2)
        pane.text(2.35, 0.512, "Prespecified limit", ax=ax, va="bottom", ha="right", color=INK2)
        pane.legend([Patch(facecolor=LIGHT, edgecolor=INK2, linewidth=0.6),
                     Patch(facecolor=MATCHED, edgecolor=MATCHED, linewidth=0.6)],
                    [EARLIER, VALUE], 2.4, 65.0)
    else:
        ax.set_ylim(-1, 6)
        pane.ticks(ax, "y", [-1, 0, 2, 4, 6])
        pane.axis_title(ax, "y", "vs routine bundle (95% CI)", labelpad=1.5)
        ax.axhline(0, linewidth=0.6, color=INK, zorder=1)


def figure4(audit):
    plate = Plate("Figure4", 180, 70, audit)
    a = plate.panel("A", Region(0, 0, 84, 70), "When to order the next test")
    b = plate.panel("B", Region(88, 0, 44, 70), "Fee relative to routine",
                    crop=Region(87.4, 0, 56.7, 70))
    c = plate.panel("C", Region(136, 0, 44, 70), crop=Region(127.6, 0, 52.4, 70))
    c.text(-7.4, 3.35, "C", size=9, bold=True)
    c.text(-2.6, 3.35, "Extra deaths not flagged per 1000", size=8, bold=True)
    order_chart(a)
    data = Table("F4BC_cost_missed.csv")
    ordering_outcomes(b, data, True)
    ordering_outcomes(c, data, False)
    plate.save()



def line_legend(names, colors, *, marker=None):
    return [Line2D([], [], color=colors[name], linewidth=1.0, marker=marker,
                   markersize=2, markeredgewidth=0) for name in names]


def calibration(pane):
    data = Table("F5A_calibration_bins.csv")
    oe = Table("F5A_OE_labels.csv")
    for i, name in enumerate(COHORTS):
        x0 = 12.1 + i * 34.0
        size = 28.7
        ax = pane.plot(x0, 11.9, size, size)
        panel_plot_style(ax)
        ax.tick_params(axis="y", pad=1)
        rows = data.rows[data.rows.cohort.eq(name)]
        maximum, maximum_row, maximum_column = max_cell(rows, ["predicted", "observed"])
        upper = maximum * 1.08
        ticks = nice_ticks(0, upper, bins=2)
        tick_labels = [data.derived([maximum_row],
                       f"{maximum_column}: ticks from 1.08 × maximum", decimal(v))
                       for v in ticks]
        ax.set(xlim=(0, upper), ylim=(0, upper))
        pane.ticks(ax, "x", ticks, tick_labels)
        pane.ticks(ax, "y", ticks, tick_labels)
        pane.axis_title(ax, "x", "Predicted", labelpad=1.3)
        if i == 0:
            pane.axis_title(ax, "y", "Observed", labelpad=1.3)
        pane.text(x0 + size / 2, 8.75, data.label(rows.iloc[0], "cohort"),
                  size=7, ha="center")
        ax.plot([0, upper], [0, upper], color=INK2, linewidth=0.6, linestyle=DASH)
        for model, color in SERIES.items():
            series = rows[rows.panel.eq(model)].sort_values("bin")
            ax.plot(series.predicted, series.observed, marker="o", markersize=2,
                    markeredgewidth=0, color=color, linewidth=1.0)
        row = find(oe.rows, cohort=name)
        pane.text(0.97, 0.041, ("O/E ", oe.label(row, "matched_OE", lambda v: f"{v:.2f}"),
                  " vs ", oe.label(row, "routine_OE", lambda v: f"{v:.2f}")),
                  ax=ax, transform=ax.transAxes, size=6, color=INK2, ha="right", va="bottom")
    names = list(SERIES)
    pane.legend(line_legend(names, SERIES, marker="o"), names, 90, 48.3, center=True)


def recalibration(pane):
    data = Table("F5B_recalibration.csv")
    rows = data.rows[data.rows.panel.eq("matched")]
    ax = pane.plot(13.0, 8.6, 107.0, 30.4)
    panel_plot_style(ax)
    positions = [0, 1, 2]
    example = rows[rows.cohort.eq(NHANES[0])].sort_values("n_local")
    pane.ticks(ax, "x", positions, [data.label(row, "n_local") for _, row in example.iterrows()])
    ticks = [0, 0.25, 0.5, 0.75, 1]
    pane.ticks(ax, "y", ticks, [percent(v) for v in ticks])
    ax.set(xlim=(-0.66, 2), ylim=(0, 1))
    pane.axis_title(ax, "x", "Local participants with known outcomes", labelpad=1.2)
    pane.axis_title(ax, "y", "Repetitions with better calibration")
    ax.yaxis.set_label_coords(-0.086, 0.44)
    for level, text in [(0.5, "50%: unnecessary below"), (0.7, "70%: threshold for helpful")]:
        ax.axhline(level, color=INK2, linewidth=0.6, linestyle=DASH, zorder=1)
        pane.text(-0.615, level + 0.022, text, ax=ax, size=6, color=INK2, va="bottom")
    endpoints = []
    for name in (NHANES[0], NHANES[2], "CHNS", NHANES[1], "Understanding Society"):
        cohort = rows[rows.cohort.eq(name)].sort_values("n_local")
        color = WARM if name in (NHANES[1], "Understanding Society") else ROUTINE
        ax.plot(positions, cohort.improved_share, color=color, linewidth=1.0, marker="o",
                markersize=3, markeredgewidth=0, clip_on=False, zorder=3)
        endpoints.append(cohort.iloc[-1])

    # Keep the endpoint labels in their plotted order when two endpoints
    # share nearly the same height.
    endpoints.sort(key=lambda row: row.improved_share)
    label_y = []
    for row in endpoints:
        label_y.append(max(row.improved_share, label_y[-1] + 0.105) if label_y else row.improved_share)
    overflow = max(0.0, label_y[-1] - 0.96)
    for y, row in zip(label_y, endpoints, strict=True):
        pane.text(1.032, y - overflow, (data.label(row, "cohort"), " (O/E ",
                  data.label(row, "import_OE", lambda v: f"{v:.2f}"), ")"),
                  ax=ax, transform=ax.get_yaxis_transform(), color=INK2)


def figure5(audit):
    plate = Plate("Figure5", 180, 100, audit)
    a = plate.panel("A", Region(0, 0, 180, 52.0), "Observed versus predicted risk")
    b = plate.panel("B", Region(0, 52.0, 180, 48.0), "Does local recalibration help?")
    calibration(a)
    recalibration(b)
    plate.save()


def appendix1(audit):
    plate = Plate("AppendixFigureS1", 180, 110, audit)
    pane = plate.panel("", Region(0, 0, 180, 110))
    data = Table("S1_subgroups.csv")
    names = (NHANES[0], NHANES[1], "Understanding Society")
    groups = list(dict.fromkeys(data.rows.group))
    y_for_group = {group: i for i, group in enumerate(groups)}
    for i, name in enumerate(names):
        x0 = 33.8 + i * 48.5
        ax = pane.plot(x0, 9.6, 40.4, 88.4)
        forest_axes(ax, (len(groups) - 0.48, -0.52), (-0.04, 0.02), -0.01)
        ax.tick_params(axis="x", pad=2)
        pane.ticks(ax, "x", [-0.04, -0.02, 0, 0.02])
        rows = data.rows[data.rows.cohort.eq(name)]
        pane.text(x0 + 20.2, 4.9, data.label(rows.iloc[0], "cohort"), size=7, bold=True, ha="center")
        if i == 0:
            labels = [data.label(data.rows[data.rows.group.eq(group)].iloc[0], "group",
                                 lambda s: s[:1].upper() + s[1:]) for group in groups]
            pane.ticks(ax, "y", list(range(len(groups))), labels)
        else:
            ax.set_yticks([])
        for _, row in rows.iterrows():
            error_point(ax, row.est, y_for_group[row.group], row.lo, row.hi, MATCHED)
    pane.text(102, 107.2, "Matched minus routine (C or AUC)", size=7.5, ha="center")
    plate.save()


@dataclass(frozen=True)
class FlowCard:
    x: float
    y: float
    width: float
    height: float
    step: Label
    count: Label


@dataclass(frozen=True)
class FlowColumn:
    name: Label
    cards: tuple
    arrows: tuple


def flow_columns(data):
    columns = []
    for i, (name, rows) in enumerate(data.rows.groupby("cohort", sort=False)):
        x = 2.4 + 25.36 * i
        cards = tuple(FlowCard(x, 15.7 + 14.75 * j, 22.25, 11.75,
                               data.label(row, "step"), data.label(row, "n", lambda n: f"{n:,}".replace(",", " ") if n >= 10000 else f"{n}"))
                      for j, (_, row) in enumerate(rows.sort_values("step_order").iterrows()))
        links = []
        for j, (first, second) in enumerate(zip(cards, cards[1:])):
            center = x + first.width / 2
            if name.startswith("CLHLS") and j == len(cards) - 2:
                # The final two rows are disjoint children of the shared
                # analyte-complete population. Route the second link in the gutter.
                parent = cards[-3]
                lane = x + first.width + 1.58
                bend = second.y - 1.55
                entry = x + first.width - 4.3
                links.append(((x + parent.width + 0.42, parent.y + parent.height / 2),
                              (lane, parent.y + parent.height / 2), (lane, bend),
                              (entry, bend), (entry, second.y - 0.47)))
            else:
                links.append(((center, first.y + first.height + 0.43),
                              (center, second.y - 0.43)))
        columns.append(FlowColumn(data.label(rows.iloc[0], "cohort"), cards, tuple(links)))
    return columns


def native_flow_pptx(columns, audit):
    try:
        from pptx import Presentation
    except ModuleNotFoundError as error:
        if error.name != "pptx":
            raise
        return None
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.oxml.xmlchemy import OxmlElement
    from pptx.util import Mm, Pt

    def rgb(value):
        return RGBColor.from_string(value.lstrip("#").upper())

    def remove_theme_effects(shape):
        style = shape._element.find(qn("p:style"))
        style.find(qn("a:effectRef")).set("idx", "0")
        shape._element.spPr.append(OxmlElement("a:effectLst"))

    def label(shape, lines, size, bold=False):
        text_frame = shape.text_frame
        text_frame.clear()
        text_frame.margin_left = text_frame.margin_right = Mm(0.75)
        text_frame.margin_top = text_frame.margin_bottom = Mm(0)
        text_frame.auto_size = MSO_AUTO_SIZE.NONE
        text_frame.word_wrap = False
        text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, line in enumerate(lines):
            paragraph = text_frame.add_paragraph() if i else text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.CENTER
            paragraph.space_before = paragraph.space_after = Pt(0)
            paragraph.line_spacing = Pt(size * 1.12)
            run = paragraph.add_run()
            run.text = line
            run.font.name = "Arial"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(INK)

    deck = Presentation()
    deck.slide_width, deck.slide_height = Mm(180), Mm(120)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    card_count = 0
    for column in columns:
        first = column.cards[0]
        heading = slide.shapes.add_textbox(Mm(first.x - 1), Mm(2), Mm(first.width + 2), Mm(11.4))
        label(heading, wrap(column.name.text, first.width + 0.4, 7, True).splitlines(), 7, True)
        for card in column.cards:
            shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Mm(card.x - 0.4), Mm(card.y - 0.4),
                                           Mm(card.width + 0.8), Mm(card.height + 0.8))
            shape.adjustments[0] = 1.5 / (card.height + 0.8)
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(LIGHT)
            shape.line.color.rgb = rgb(INK2)
            shape.line.width = Pt(0.6)
            remove_theme_effects(shape)
            label(shape, wrap(card.step.text, card.width - 1.8, 6.5).splitlines() + [card.count.text], 6.5)
            card_count += 1

        for path in column.arrows:
            for j, (start, end) in enumerate(zip(path, path[1:])):
                connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Mm(start[0]), Mm(start[1]),
                                                       Mm(end[0]), Mm(end[1]))
                connector.line.color.rgb = rgb(INK2)
                connector.line.width = Pt(0.8)
                remove_theme_effects(connector)
                if j == len(path) - 2:
                    end_tag = OxmlElement("a:tailEnd")
                    end_tag.set("type", "triangle")
                    end_tag.set("w", "sm")
                    end_tag.set("len", "sm")
                    connector.line._get_or_add_ln().append(end_tag)

    target = OUT / "AppendixFigureS2.pptx"
    deck.save(str(target))
    reopened = Presentation(str(target))
    assert len(reopened.slides) == 1
    with ZipFile(target) as archive:
        xml = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    dml = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    pml = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
    audit.check_text("AppendixFigureS2", [node.text or "" for node in xml.iter(dml + "t")])
    assert len(list(xml.iter(pml + "sp"))) == card_count + len(columns)
    assert not list(xml.iter(pml + "pic"))
    assert all(effect.get("idx") == "0" for effect in xml.iter(dml + "effectRef"))
    assert len(list(xml.iter(dml + "tailEnd"))) == sum(len(c.arrows) for c in columns)
    return card_count


def appendix2(audit):
    plate = Plate("AppendixFigureS2", 180, 120, audit)
    pane = plate.panel("", Region(0, 0, 180, 120))
    columns = flow_columns(Table("S2_flow.csv"))
    for column in columns:
        first = column.cards[0]
        pane.text(first.x + first.width / 2, 7.7, column.name, size=7, bold=True,
                  ha="center", max_width=first.width + 0.4)
        for card in column.cards:
            box(pane.base, card.x, card.y, card.width, card.height, LIGHT, INK2)
            pane.text(card.x + card.width / 2, card.y + card.height / 2,
                      (card.step, "\n", card.count), ha="center", max_width=card.width - 1.8)
        for path in column.arrows:
            arrow(pane.base, path[0], path[-1], scale=4.9, path=path if len(path) > 2 else None)
    plate.save()
    return native_flow_pptx(columns, audit)



def appendix3(audit):
    plate = Plate("AppendixFigureS3", 180, 70, audit)
    pane = plate.panel("", Region(0, 0, 180, 70))
    data = Table("S3_decision_curves.csv")
    visible = data.rows[
        (data.rows.cohort.eq(NHANES[0]) & data.rows.threshold.le(0.30)) |
        (data.rows.cohort.eq(NHANES[1]) & data.rows.threshold.le(0.15))
    ]
    maximum, max_row, max_column = max_cell(visible, ["net_benefit"])
    upper = math.ceil(maximum * 100) / 100 + 0.01
    yticks = nice_ticks(-0.02, upper, bins=4)
    ylabels = [data.derived([max_row], f"{max_column}: ticks from shared y limit", decimal(t))
               for t in yticks]
    for i, (name, upper_x, threshold) in enumerate([(NHANES[0], 0.30, 0.10),
                                                     (NHANES[1], 0.15, 0.05)]):
        x0 = 13.4 + i * 85.8
        ax = pane.plot(x0, 12.5, 73.8, 40.0)
        panel_plot_style(ax)
        rows = visible[visible.cohort.eq(name)]
        pane.text(x0 + 36.9, 5.05, data.label(rows.iloc[0], "cohort"),
                  size=7, bold=True, ha="center")
        ax.set(xlim=(0.01, upper_x), ylim=(-0.02, upper))
        xticks = [float(v) for v in np.arange(0.05, upper_x + 1e-9, 0.05)]
        pane.ticks(ax, "x", xticks, [percent(v) for v in xticks])
        pane.ticks(ax, "y", yticks, ylabels)
        pane.axis_title(ax, "x", "Threshold probability", labelpad=2.0)
        if i == 0:
            pane.axis_title(ax, "y", "Net benefit", labelpad=2.0)
        ax.axhline(0, color=INK, linewidth=0.6, zorder=1)
        ax.axvline(threshold, color=INK2, linewidth=0.6, linestyle=DASH, zorder=1)
        pane.text(threshold, 1.023, "Prespecified threshold", ax=ax, transform=ax.get_xaxis_transform(),
                  color=INK2, va="bottom", ha="center")
        for name, color in {**SERIES, "Treat all": INK2}.items():
            series = rows[rows.strategy.eq(name)].sort_values("threshold")
            ax.plot(series.threshold, series.net_benefit, color=color, linewidth=1.0,
                    linestyle=DASH if name == "Treat all" else "-")
    names = [*SERIES, "Treat all"]
    handles = line_legend(list(SERIES), SERIES) + [Line2D([], [], color=INK2, linestyle=DASH, linewidth=1.0)]
    pane.legend(handles, names, 90, 66.4, center=True)
    plate.save()


BENCHMARK_LABELS = {
    "E1": "FPMTB (drug sensitivity)", "E2": "PACE (drug sensitivity)",
    "E3": "TCGA breast (proteomics)", "E4": "GTEx blood (expression)",
    "Beat AML": "Beat AML (drug sensitivity)", "PKIS2": "PKIS2 (kinase profiles)",
    "M": "NHANES 2015–16 (set 1)", "A1": "NHANES 2015–16 (set 2)",
    "A2": "NHANES 2013–14", "A3": "NHANES 2011–12",
}


def benchmark_dumbbells(pane, data, rows):
    ax = pane.plot(37.1, 8.4, 57.2, 48.5)
    panel_plot_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    maximum, max_row, max_column = max_cell(rows, ["reuse_ev", "aware_ev"])
    upper = math.ceil(maximum * 10) / 10
    ax.set(xlim=(0, upper), ylim=(len(rows) - 0.48, -0.52))
    xticks = nice_ticks(0, upper, bins=4)
    pane.ticks(ax, "x", xticks, [data.derived([max_row],
               f"{max_column}: ticks from x limit", decimal(t)) for t in xticks])
    ylabels = [data.derived([row.name], "source: brief label mapping", BENCHMARK_LABELS[row.source])
               for _, row in rows.iterrows()]
    pane.ticks(ax, "y", list(range(len(rows))), ylabels)
    pane.axis_title(ax, "x", "Explained variance of held-out targets", labelpad=2)
    for y, (_, row) in enumerate(rows.iterrows()):
        ax.plot([row.reuse_ev, row.aware_ev], [y, y], color=INK2, linewidth=1.0)
        ax.plot(row.reuse_ev, y, "o", markersize=4, color=ROUTINE, markeredgewidth=0)
        ax.plot(row.aware_ev, y, "o", markersize=4, color=MATCHED, markeredgewidth=0)
    handles = [Line2D([], [], marker="o", markersize=4, markeredgewidth=0, color=color,
                      linestyle="none") for color in (ROUTINE, MATCHED)]
    pane.legend(handles, ["Best reusable panel", "Target-aware panel"], 65.7, 70.4, center=True)


def benchmark_leads(pane, data):
    ax = pane.plot(14.3, 8.4, 61.1, 48.5)
    panel_plot_style(ax)
    ax.set_xlim(-0.43, 3.43)
    maximum, max_row, max_column = max_cell(data.rows, ["aware_minus_reuse"])
    upper = math.ceil(maximum * 20) / 20
    ax.set_ylim(-0.01, upper)
    yticks = nice_ticks(0, upper, bins=4)
    pane.ticks(ax, "y", yticks, [data.derived([max_row],
               f"{max_column}: ticks from y limit", decimal(t)) for t in yticks])
    pane.axis_title(ax, "y", "Lead over best reusable panel", labelpad=2)
    ax.axhline(0, linewidth=0.6, color=INK)
    ax.axhline(0.02, linewidth=0.6, color=INK2, linestyle=DASH)
    pane.text(1.021, 0.02, "0.02", ax=ax, transform=ax.get_yaxis_transform(), color=INK2)
    rng = np.random.default_rng(1)
    labels = []
    for i, (k, predictor) in enumerate([(5, "ridge"), (5, "gbt"), (10, "ridge"), (10, "gbt")]):
        rows = data.rows[data.rows.k.eq(k) & data.rows.predictor.eq(predictor)]
        first = rows.iloc[0]
        labels.append((data.label(first, "k"), " variables\n", data.label(
            first, "predictor", lambda name: "tree-based" if name == "gbt" else "linear")))
        ax.plot(i + rng.uniform(-0.15, 0.15, len(rows)), rows.aware_minus_reuse,
                linestyle="none", marker="o", markersize=3.4, color=MATCHED, markeredgewidth=0)
    pane.ticks(ax, "x", list(range(len(labels))), labels)


def appendix4(audit):
    plate = Plate("AppendixFigureS4", 180, 75, audit)
    data = Table("S4_benchmark.csv")
    rows = data.rows[data.rows.k.eq(5) & data.rows.predictor.eq("ridge")].sort_values("aware_ev")
    title = (data.label(rows.iloc[0], "k"), " measured variables, ",
             data.label(rows.iloc[0], "predictor", lambda name: "linear model" if name == "ridge" else name))
    a = plate.panel("A", Region(0, 0, 98, 75), title)
    b = plate.panel("B", Region(98, 0, 82, 75), "Target-aware lead in every setting")
    benchmark_dumbbells(a, data, rows)
    benchmark_leads(b, data)
    plate.save()


def appendix5(audit):
    plate = Plate("AppendixFigureS5", 120, 78, audit)
    pane = plate.panel("", Region(0, 0, 120, 78), "NHANES 2011–14: fee and discrimination")
    scatter_cost(pane, Table("S5_cost_C_NHANES2011_14.csv"), 5, (0.71, 0.82))
    plate.save()


FIGURE_SIZES = {
    "Figure1": (180, 125), "Figure2": (180, 150), "Figure3": (180, 100),
    "Figure4": (180, 70), "Figure5": (180, 100),
    "AppendixFigureS1": (180, 110), "AppendixFigureS2": (180, 120),
    "AppendixFigureS3": (180, 70), "AppendixFigureS4": (180, 75),
}
FIGURE_PARTS = {
    "Figure1": "ABCD", "Figure2": "ABC", "Figure3": "ABC", "Figure4": "ABC",
    "Figure5": "AB", "AppendixFigureS4": "AB",
}


def verify_outputs():
    mismatches = []
    for name, size in FIGURE_SIZES.items():
        for suffix in ("pdf", "svg", "png"):
            assert (OUT / f"{name}.{suffix}").stat().st_size > 0
        svg = ET.parse(OUT / f"{name}.svg").getroot()
        physical = tuple(float(svg.attrib[key].removesuffix("pt")) * 25.4 / 72
                         for key in ("width", "height"))
        if any(abs(want - got) > 0.06 for want, got in zip(size, physical)):
            mismatches.append((name, size, tuple(round(v, 2) for v in physical)))
        for panel in FIGURE_PARTS.get(name, ""):
            assert (PANELS / f"{name}_{panel}.pdf").stat().st_size > 0
            root = ET.parse(PANELS / f"{name}_{panel}.svg").getroot()
            assert list(root.iter("{http://www.w3.org/2000/svg}text"))
    if mismatches:
        print("Physical sizes requiring review:", mismatches)
    return mismatches


def write_readme(audit, card_count, mismatches):
    lines = [
        "# Figures", "",
        "Generated by `make_figures_v2.py` from the CSV files in `data/` (written by `export_figure_data.py`).", "",
        "| Figure | Combined | Separate parts |",
        "| --- | --- | --- |",
    ]
    for name in FIGURE_SIZES:
        combined = ", ".join(f"{name}.{ext}" for ext in ("pdf", "svg", "png"))
        parts = ", ".join(f"panels/{name}_{p}.pdf/.svg" for p in FIGURE_PARTS.get(name, ""))
        lines.append(f"| {name} | {combined} | {parts or '-'} |")
    lines.extend([
        "",
        ("- AppendixFigureS2.pptx: editable flow diagram (native text boxes and connectors)." if card_count is not None else
         "- python-pptx not installed: AppendixFigureS2.pptx was not written."),
        f"- figure_numbers_used.csv: every printed number ({len(audit.rows)}) with its source file, row and column; "
        "the script checks that each number printed in the SVG files matches this register.",
        f"- Physical sizes (mm): {'to review: ' + str(mismatches) if mismatches else 'as specified.'}",
        "",
    ])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    PANELS.mkdir(exist_ok=True)
    audit = Audit()
    for draw in (figure1, figure2, figure3, figure4, figure5, appendix1):
        draw(audit)
    card_count = appendix2(audit)
    for draw in (appendix3, appendix4):
        draw(audit)
    audit.save()
    mismatches = verify_outputs()
    write_readme(audit, card_count, mismatches)
    print(f"Created {len(FIGURE_SIZES)} combined figures and "
          f"{sum(map(len, FIGURE_PARTS.values()))} panel pairs. "
          f"Verified {len(audit.rows)} printed numeric tokens.")
    if mismatches:
        raise ValueError("Combined export sizes differ from the brief")


if __name__ == "__main__":
    try:
        main()
    finally:
        plt.close("all")
        PLOT_CACHE.cleanup()
