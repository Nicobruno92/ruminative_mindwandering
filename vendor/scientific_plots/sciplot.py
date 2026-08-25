"""
sciplot — publication figure toolkit (Plotly-first).

Handles the three things that are easy to get wrong and boring to redo:

1. **Physical sizing.** A journal figure is specified in millimetres, not
   pixels. Plotly font sizes are in CSS px. This module converts, so that
   asking for 8 pt text in a 89 mm figure actually yields 8 pt text on paper.
2. **A shared look.** One template + one palette file, so every figure of a
   project (and every panel of a figure, including matplotlib ones) matches.
3. **Export.** SVG (vector, editable) + PNG (raster preview) at a controlled
   effective DPI, from a single call, guaranteed to be the same figure.

Usage
-----
    import sys; sys.path.insert(0, "<skill>/scripts")
    import sciplot as sp
    import plotly.graph_objects as go

    pal = sp.init(__file__)                     # palette + template
    fig = go.Figure(go.Bar(x=..., y=..., marker_color=pal.get("control")))
    sp.save(fig, "figures/fig1", "fig1_behavior", width_mm=89, height_mm=58)

Every public function is a plain function on a plain plotly Figure — nothing is
hidden, and anything the template does can be overridden afterwards with the
normal `fig.update_layout(...)` API.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import plotly.graph_objects as go
import plotly.io as pio
import yaml

# =============================================================================
# Units
# =============================================================================
# An SVG written by plotly carries width/height in CSS pixels with no unit, so
# downstream tools (Inkscape, Illustrator, LaTeX) read 1 px = 1/96 inch.
# Therefore:  mm -> px is a fixed conversion, and a font size given in px
# lands on paper as px * 72/96 points.
MM_PER_IN = 25.4
CSS_DPI = 96.0

#: Journal column widths in mm. Nature/Science/Cell agree to within ~2 mm;
#: these are safe defaults. Check the target journal before submission.
WIDTHS_MM = {
    "single": 89.0,    # one column
    "onehalf": 120.0,  # 1.5 columns
    "double": 180.0,   # full page width
}

#: Default type sizes in POINTS at final print size. Journals accept 5-7 pt as
#: an absolute floor read at 100% zoom on a printed page. This project's
#: figures are embedded in a paper PDF and then routinely viewed *smaller*
#: than that (a PDF window well short of full-screen, a slide thumbnail) —
#: so the house floor sits above the journal minimum on purpose: 9-10 pt
#: is the size that still reads once the figure has been shrunk again on
#: top of its own print-size shrink. Do not "fix" small text downstream
#: (bumping one call's font dict) — raise the house floor here so every
#: figure inherits it, and keep checking that panels/whitespace shrank
#: to make room rather than leaving type as the smallest element on the
#: page (see the canvas-size rule in SKILL.md).
TYPE_PT = {
    "axis_title": 10.0,
    "tick": 9.0,
    "legend": 9.0,
    "annotation": 9.0,
    "panel_label": 12.0,
}

#: PNG raster resolution. 600 is the project's minimum for a figure that
#: ships in a paper — dpi only changes PNG pixel density, never layout (SVG
#: is vector regardless, and mm/pt sizing is dpi-independent), so it is safe
#: to render cheap low-dpi copies while iterating and only pay for 600 dpi on
#: the final call. Pass an explicit lower ``dpi=`` (100-150) during the
#: look-fix-look loop; leave it at the default for the one delivered PNG.
DEFAULT_DPI = 600.0
FONT_STACK = "Helvetica Neue, Helvetica, Arial, Nimbus Sans, DejaVu Sans, sans-serif"

_ASSET_PALETTE = Path(__file__).resolve().parent.parent / "assets" / "color_palette.default.yaml"


def mm2px(mm: float) -> float:
    """Millimetres to CSS pixels (the unit plotly lays out in)."""
    return mm / MM_PER_IN * CSS_DPI


def px2mm(px: float) -> float:
    """CSS pixels to millimetres."""
    return px / CSS_DPI * MM_PER_IN


def pt2px(pt: float) -> float:
    """Points (final print size) to the px number plotly expects."""
    return pt * CSS_DPI / 72.0


def px2pt(px: float) -> float:
    """Px in a plotly/SVG document to points on paper."""
    return px * 72.0 / CSS_DPI


# =============================================================================
# Palette
# =============================================================================
class Palette:
    """A project's color decisions, backed by ``color_palette.yaml``.

    Resolution order for :meth:`get` is: explicit per-category assignment ->
    named swatch of the active cycle -> neutral/semantic color -> literal hex.
    Unknown names raise, because silently returning a default color is how a
    figure ends up with two groups sharing a hue.
    """

    def __init__(self, path: Path, data: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.data = dict(data)
        active = self.data.get("active", "observable10")
        cycles = self.data.get("available", {})
        if active not in cycles:
            raise KeyError(
                f"active palette {active!r} not found in {self.path}; "
                f"available: {sorted(cycles)}"
            )
        self.name = active
        self.order: list[str] = list(cycles[active]["order"])
        self.named: dict[str, str] = dict(cycles[active].get("named", {}))
        self.neutral: dict[str, str] = dict(self.data.get("neutral", {}))
        self.continuous: dict[str, str] = dict(self.data.get("continuous", {}))
        self.categories: dict[str, str] = dict(self.data.get("categories") or {})

    # -- lookup ------------------------------------------------------------
    def get(self, key: str) -> str:
        """Resolve a color name (category, swatch, neutral, or literal hex)."""
        if isinstance(key, str) and key.startswith("#"):
            return key
        for table in (self.categories, self.named, self.neutral):
            if key in table:
                return table[key]
        raise KeyError(
            f"{key!r} is not a known color. Known categories: "
            f"{sorted(self.categories)}; swatches: {sorted(self.named)}; "
            f"neutrals: {sorted(self.neutral)}. "
            f"Assign it with pal.map([...]) or edit {self.path}."
        )

    def cycle(self, n: int) -> list[str]:
        """First ``n`` colors of the active cycle (wraps if n > cycle length)."""
        return [self.order[i % len(self.order)] for i in range(n)]

    def map(self, values: Iterable[Any], *, persist: bool = True) -> dict[str, str]:
        """Assign a stable color to each category, reusing existing decisions.

        New categories take the next unused colors of the cycle and are written
        back to ``color_palette.yaml`` so the same group keeps the same color in
        every later figure. Report the new assignments to the user — they are a
        project-level decision, not a per-figure one.
        """
        labels = [str(v) for v in values]
        seen: list[str] = []
        for label in labels:  # dedupe, preserve order
            if label not in seen:
                seen.append(label)

        used = set(self.categories.values())
        free = [c for c in self.order if c not in used] or list(self.order)
        assigned: dict[str, str] = {}
        new: dict[str, str] = {}
        k = 0
        for label in seen:
            if label in self.categories:
                assigned[label] = self.categories[label]
            else:
                color = free[k % len(free)]
                k += 1
                assigned[label] = color
                new[label] = color
                self.categories[label] = color

        if new and persist:
            _append_categories(self.path, new)
        self.last_new_assignments = new
        return assigned

    def colorscale(self, kind: str = "sequential") -> str:
        """Named plotly colorscale for ``sequential`` / ``diverging`` data."""
        if kind not in self.continuous:
            raise KeyError(f"{kind!r} not in continuous: {sorted(self.continuous)}")
        return self.continuous[kind]


def find_palette(anchor: str | Path | None = None) -> Path | None:
    """Walk up from ``anchor`` looking for an existing ``color_palette.yaml``.

    Stops at the repository root (a directory containing ``.git``) so a stray
    palette from a parent project is never picked up by accident.
    """
    start = Path(anchor).resolve() if anchor else Path.cwd().resolve()
    if start.is_file():
        start = start.parent
    for d in [start, *start.parents]:
        candidate = d / "color_palette.yaml"
        if candidate.exists():
            return candidate
        if (d / ".git").exists():
            return None
    return None


def repo_root(anchor: str | Path | None = None) -> Path:
    """Nearest directory containing ``.git``; falls back to ``anchor`` itself."""
    start = Path(anchor).resolve() if anchor else Path.cwd().resolve()
    if start.is_file():
        start = start.parent
    for d in [start, *start.parents]:
        if (d / ".git").exists():
            return d
    return start


def load_palette(anchor: str | Path | None = None, *, create: bool = True) -> Palette:
    """Find (or create) the project palette and return it."""
    path = find_palette(anchor)
    if path is None:
        if not create:
            raise FileNotFoundError("no color_palette.yaml found")
        path = repo_root(anchor) / "color_palette.yaml"
        shutil.copyfile(_ASSET_PALETTE, path)
        print(f"[sciplot] created {path} (default palette). Tell the user; they may want to edit it.")
    return Palette(path, yaml.safe_load(path.read_text()) or {})


def _append_categories(path: Path, new: Mapping[str, str]) -> None:
    """Insert entries under the ``categories:`` key, preserving comments.

    A YAML round-trip through ``safe_load``/``safe_dump`` would strip every
    comment from the palette file, and those comments are what make it editable
    by a human. So this edits the text directly.
    """
    lines = path.read_text().splitlines()
    block = "\n".join(f'  {k}: "{v}"' for k, v in new.items())

    idx = next((i for i, ln in enumerate(lines) if re.match(r"^categories:\s*(\{\s*\}\s*)?$", ln)), None)
    if idx is None:
        lines += ["", "categories:", *block.splitlines()]
    else:
        lines[idx] = "categories:"  # drop a literal `{}` if present
        end = idx + 1
        while end < len(lines) and (lines[end].startswith("  ") or not lines[end].strip()):
            end += 1
        while end > idx + 1 and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = block.splitlines()
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# Template
# =============================================================================
def make_template(pal: Palette, *, type_pt: Mapping[str, float] | None = None) -> go.layout.Template:
    """Build the ``sci`` template: plotly_white, tightened and de-cluttered.

    The differences from ``plotly_white`` are all about ink and space:
    gridlines off by default (turn them back on per-axis with :func:`grid` only
    where a reader has to compare distant values), a real axis line and outside
    ticks so the data region is defined without a box, no title, tiny margins
    with ``automargin`` doing the work, the project colorway, and the house
    type sizes in ``TYPE_PT`` (10 pt axis titles / 9 pt ticks by default —
    above the journal floor, see ``TYPE_PT``'s docstring for why).
    """
    t = dict(TYPE_PT)
    t.update(type_pt or {})
    ink = pal.neutral.get("ink", "#2B2B2B")
    grid_c = pal.neutral.get("grid", "#E9E9E9")

    axis = dict(
        showline=True,
        linecolor=ink,
        linewidth=0.8,
        ticks="outside",
        tickcolor=ink,
        ticklen=3,
        tickwidth=0.8,
        showgrid=False,
        zeroline=False,
        gridcolor=grid_c,
        gridwidth=0.7,
        automargin=True,
        title=dict(font=dict(size=pt2px(t["axis_title"]), color=ink), standoff=4),
        tickfont=dict(size=pt2px(t["tick"]), color=ink),
    )

    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_STACK, size=pt2px(t["tick"]), color=ink),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=pal.order,
        title=None,
        margin=dict(l=4, r=6, t=6, b=4, pad=0),
        xaxis=axis,
        yaxis=axis,
        legend=dict(
            font=dict(size=pt2px(t["legend"]), color=ink),
            # Opaque enough to stay readable when parked over data, but note
            # that a legend sitting on top of data is a defect to fix, not to
            # paper over — see legend_inside().
            bgcolor="rgba(255,255,255,0.88)",
            borderwidth=0,
            tracegroupgap=3,
            # itemsizing is left at "trace" on purpose: "constant" draws a 5 px
            # legend line regardless of the trace's own width, so the legend
            # stops matching the lines it describes.
        ),
        colorscale=dict(
            sequential=pal.continuous.get("sequential", "Viridis"),
            diverging=pal.continuous.get("diverging", "RdBu"),
        ),
        coloraxis=dict(
            colorbar=dict(
                outlinewidth=0,
                thickness=6,
                len=0.75,
                ticks="outside",
                ticklen=2,
                tickwidth=0.7,
                tickfont=dict(size=pt2px(t["tick"])),
                title=dict(font=dict(size=pt2px(t["axis_title"]))),
            )
        ),
        annotationdefaults=dict(
            font=dict(size=pt2px(t["annotation"]), color=ink),
            showarrow=False,
        ),
        hoverlabel=dict(font=dict(family=FONT_STACK)),
    )
    template.data.scatter = [go.Scatter(line=dict(width=1.4), marker=dict(size=5, line=dict(width=0)))]
    template.data.scattergl = [go.Scattergl(line=dict(width=1.4), marker=dict(size=5))]
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    template.data.box = [go.Box(line=dict(width=1.0), marker=dict(size=3))]
    template.data.violin = [go.Violin(line=dict(width=1.0))]
    template.data.heatmap = [go.Heatmap(colorbar=dict(outlinewidth=0, thickness=6))]
    return template


def init(anchor: str | Path | None = None, *, type_pt: Mapping[str, float] | None = None) -> Palette:
    """Load the palette, register the ``sci`` template and make it the default.

    Call once at the top of a figure script, passing ``__file__`` so the palette
    is resolved relative to the script rather than the current directory.
    """
    pal = load_palette(anchor)
    pio.templates["sci"] = make_template(pal, type_pt=type_pt)
    pio.templates.default = "sci"
    return pal


# =============================================================================
# Layout helpers
# =============================================================================
def grid(fig: go.Figure, axis: str = "y", **kwargs: Any) -> go.Figure:
    """Switch gridlines on for one axis (all subplots).

    Gridlines are off by default because most figures do not need them. Turn on
    the axis that carries the quantity a reader must compare across distant
    categories — never both, and never a grid denser than the ticks.
    """
    opts = dict(showgrid=True, **kwargs)
    if axis in ("x", "both"):
        fig.update_xaxes(**opts)
    if axis in ("y", "both"):
        fig.update_yaxes(**opts)
    return fig


def legend_inside(
    fig: go.Figure,
    corner: str = "top right",
    *,
    orientation: str = "v",
    pad: float = 0.01,
) -> go.Figure:
    """Park the legend in empty space inside the axes.

    A legend outside the plot area eats 15-25% of the figure width for a few
    words. Whenever the data leaves a corner empty, put the legend there and
    give that width back to the data.
    """
    v, h = (corner.split() + ["right"])[:2]
    y, yanchor = (1 - pad, "top") if v == "top" else (pad, "bottom")
    x, xanchor = (1 - pad, "right") if h == "right" else (pad, "left")
    fig.update_layout(
        legend=dict(orientation=orientation, x=x, y=y, xanchor=xanchor, yanchor=yanchor)
    )
    return fig


def legend_above(fig: go.Figure, *, x: float = 0.0, anchor: str = "left") -> go.Figure:
    """Horizontal legend in a thin strip above the axes.

    The fallback when no corner inside the plot is free: it costs one text line
    of height instead of a whole column of width.
    """
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor=anchor, x=x)
    )
    return fig


def panel_labels(
    fig: go.Figure,
    labels: Sequence[str] | None = None,
    *,
    dx: float = -0.045,
    size_pt: float = TYPE_PT["panel_label"],
) -> go.Figure:
    """Add bold A, B, C... above the top-left corner of each subplot.

    Panel labels are the one piece of text a multi-panel figure always needs:
    they are what the caption points at. Everything else should be visual.

    They are anchored *above* each subplot (not inside it) so they can never
    land on the data, and the top margin is opened just enough to hold them.
    """
    axes = sorted(
        (k for k in fig.layout if re.fullmatch(r"xaxis\d*", k)),
        key=lambda k: int(k[5:] or 1),
    )
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(axes))]
    for name, label in zip(axes, labels):
        if not label:
            continue
        xa = fig.layout[name]
        ya = fig.layout["yaxis" + name[5:]]
        fig.add_annotation(
            x=max(0.0, xa.domain[0] + dx),
            y=ya.domain[1],
            xref="paper",
            yref="paper",
            text=f"<b>{label}</b>",
            font=dict(size=pt2px(size_pt)),
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
        )
    # Reserve room for the labels; without this they are clipped by the canvas.
    need = pt2px(size_pt) * 1.25
    fig.update_layout(margin=dict(t=max(fig.layout.margin.t or 0, need)))
    return fig


# =============================================================================
# Faceting
# =============================================================================
def facet_subplots(
    rows: int,
    cols: int,
    *,
    subplot_titles: Sequence[str] | None = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    spacing_mm: float = 3.0,
    width_mm: float = WIDTHS_MM["double"],
    height_mm: float | None = None,
    **kwargs: Any,
) -> go.Figure:
    """``make_subplots`` sized so facets read as one figure, not N pasted together.

    Plotly's own defaults (``horizontal_spacing=0.2/cols``,
    ``vertical_spacing=0.3/rows``) and the "0.06-0.10" rule of thumb some
    figures use are both *fixed fractions of the whole canvas regardless of
    panel count* — a row of five panels loses four gaps of that size, which
    reads as five separate plots pasted onto one canvas rather than one
    faceted figure. This instead fixes the gutter in millimetres (a few mm
    reads as "next to each other" no matter how many panels there are) and
    converts to the fraction ``make_subplots`` wants from the figure's actual
    physical size, so adding a panel shrinks the panels a little instead of
    growing the gutter.

    Pass the same ``width_mm``/``height_mm`` you intend to use in the later
    :func:`save` call — the conversion is exact only then; a mismatch just
    skews the gutter-to-panel ratio a bit, it does not break the figure.

    ``shared_xaxes``/``shared_yaxes`` forward to ``make_subplots``. When set,
    this also hides the duplicated tick labels on interior panels (only the
    left column keeps y labels, only the bottom row keeps x labels) — a
    repeated axis reserving its own margin on every panel is the other common
    source of apparent dead space between facets. Pass ``yaxis_title`` /
    ``xaxis_title`` here (instead of calling ``fig.update_yaxes(title_text=)``
    yourself afterwards) so the single real title lands on the left/bottom
    edge *before* the interior duplicates are stripped — set it after this
    call and it re-applies to every panel, since ``update_yaxes()`` with no
    ``col`` touches all of them. The one surviving title still goes through
    the template's normal ``automargin``, so it is never clipped the way a
    hand-placed annotation can be.
    """
    from plotly.subplots import make_subplots

    height_mm = height_mm if height_mm is not None else width_mm * 0.62
    h_spacing = (spacing_mm / width_mm) if cols > 1 else 0.0
    v_spacing = (spacing_mm / height_mm) if rows > 1 else 0.0

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
        **kwargs,
    )

    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title)
    if xaxis_title:
        fig.update_xaxes(title_text=xaxis_title)

    # title_text=None means "leave alone" to plotly's update_*, not "clear", and
    # col=/row= only reliably selects one subplot at a time (a list silently
    # drops the title update) — so this clears each interior panel individually.
    if shared_yaxes and cols > 1:
        for c in range(2, cols + 1):
            fig.update_yaxes(showticklabels=False, title_text="", col=c)
    if shared_xaxes and rows > 1:
        for r in range(1, rows):
            fig.update_xaxes(showticklabels=False, title_text="", row=r)

    # subplot_titles land as annotations at the default (larger) title size;
    # shrink to the project's annotation size so they don't reopen the gap
    # this function just closed.
    for ann in fig.layout.annotations:
        ann.font = dict(size=pt2px(TYPE_PT["annotation"]))
        ann.yshift = 2

    return fig


def hollow(color: str, *, width: float = 1.2) -> dict[str, Any]:
    """Marker dict for a non-significant / de-emphasized point.

    Pairs with a filled marker of the same color: hue stays free to encode the
    variable, fill encodes significance.
    """
    return dict(color="white", line=dict(color=color, width=width))


# =============================================================================
# Export
# =============================================================================
#: Sentinel for save()/save_mpl()'s ``data`` argument: "yes, I checked, this
#: figure genuinely has no backing table" (a schematic/diagram, or one drawn
#: from a table already written by an earlier :func:`save_data` call in the
#: same script). Passing the bare Python ``None`` means the same thing but
#: reads, at the call site, exactly like "I forgot this argument" — using the
#: named sentinel instead is what makes the omission visibly deliberate to
#: the next person (or the next session) reading the script.
NO_BACKING_TABLE = object()


def save(
    fig: go.Figure,
    outdir: str | Path,
    name: str,
    *,
    width_mm: float,
    height_mm: float | None = None,
    dpi: float = DEFAULT_DPI,
    formats: Sequence[str] = ("svg", "png"),
    aspect: float = 0.62,
    data: Any | Mapping[str, Any] | None,
) -> dict[str, Path]:
    """Write the figure at a real physical size, as SVG and PNG.

    ``width_mm`` is the width the figure will occupy in the paper (89 mm for a
    single column, 180 mm for a double). Sizing here rather than resizing later
    is what keeps the house 10 pt text actually 10 pt: scaling an SVG
    afterwards scales the type with it.

    ``dpi`` defaults to 600 — the project minimum for a delivered PNG. That
    default is only for the *final* call: dpi is purely a raster-density knob
    (SVG is vector regardless, and the mm/pt layout does not change with it),
    so pass an explicit lower ``dpi=`` (100-150) for the cheap renders in a
    look-fix-look loop and only let the last call use the 600 dpi default.

    ``data`` is a **required** keyword, not an optional nicety: every figure
    needs a table someone can reopen without rerunning the pipeline, and
    making the argument optional was exactly how that stopped happening in
    practice (a call site that never got a `data=` added, quietly, forever).
    Pass the already-aggregated table(s) the figure was built from — see
    :func:`save_data` for the accepted shapes — and this writes the CSV
    backup alongside the image files in the same call. Pass
    ``sciplot.NO_BACKING_TABLE`` (not bare ``None`` — see its docstring) only
    when there genuinely is no table: a schematic/diagram, or a figure whose
    data was already written by a separate :func:`save_data` call earlier in
    the same script. That still prints a line confirming the skip was
    deliberate, so it shows up in the run log rather than passing silently.

    Returns a dict of format -> path (plus ``"csv:*"`` keys when ``data`` is
    given).
    """
    if not 30 <= width_mm <= 300:
        raise ValueError(f"width_mm={width_mm} is outside 30-300 mm; check the units")
    height_mm = height_mm if height_mm is not None else width_mm * aspect

    w_px, h_px = round(mm2px(width_mm)), round(mm2px(height_mm))
    fig.update_layout(width=w_px, height=h_px)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scale = dpi / CSS_DPI
    written: dict[str, Path] = {}
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.write_image(
            path, format=fmt, width=w_px, height=h_px, scale=scale if fmt == "png" else 1
        )
        written[fmt] = path

    if data is NO_BACKING_TABLE or data is None:
        print(f"[sciplot] {name}: no CSV backup — data={'NO_BACKING_TABLE' if data is NO_BACKING_TABLE else 'None'} "
              f"was passed explicitly to save(). Only correct for a schematic/diagram or a table "
              f"already written by a separate save_data() call.")
    else:
        written.update(save_data(outdir, name, data))

    print(
        f"[sciplot] {name}: {width_mm:.0f}x{height_mm:.0f} mm "
        f"({w_px}x{h_px} px), PNG at {dpi:.0f} dpi -> "
        + ", ".join(str(p) for p in written.values())
    )
    return written


def save_data(outdir: str | Path, name: str, data: Any | Mapping[str, Any]) -> dict[str, Path]:
    """Write the exact table(s) a figure was built from next to it, as CSV.

    Every figure needs a backing table someone can reopen without rerunning
    the pipeline — for a reviewer question, a caption number check, or a
    retouch months later. Pass whatever was already computed for plotting:

    - a single ``DataFrame`` -> ``{name}.csv``
    - ``{"panel_a": df_a, "panel_b": df_b}`` for a multi-panel figure with one
      table per panel -> ``{name}__panel_a.csv``, ``{name}__panel_b.csv``

    Anything with a ``.to_csv(path, index=False)`` method works; this does
    not require pandas specifically.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    items = data.items() if isinstance(data, Mapping) else [(None, data)]
    written: dict[str, Path] = {}
    for key, table in items:
        path = outdir / (f"{name}.csv" if key is None else f"{name}__{key}.csv")
        table.to_csv(path, index=False)
        written[f"csv:{key or name}"] = path
    return written


# =============================================================================
# matplotlib bridge
# =============================================================================
def mpl_rc(pal: Palette, *, type_pt: Mapping[str, float] | None = None) -> dict[str, Any]:
    """rcParams that make a matplotlib panel look like a plotly ``sci`` panel.

    Needed whenever one panel has to come from a domain library that only emits
    matplotlib (MNE topographies, nilearn brains, statsmodels diagnostics). A
    composed figure where panel B has different type and line weights than
    panel A reads as sloppy even when both panels are individually fine.

    Use with ``matplotlib.pyplot.rc_context(sp.mpl_rc(pal))`` and size the
    figure in inches as ``width_mm / 25.4``.

    Note that ``savefig.bbox`` is deliberately left at ``standard``. The usual
    ``bbox_inches="tight"`` crops the canvas to the ink, which silently changes
    the figure's physical width — so the panel then gets rescaled during
    composition and its type ends up a different size from every other panel.
    Use ``fig.tight_layout()`` or ``constrained_layout=True`` instead: those
    rearrange the axes *inside* a canvas of the size you asked for.
    """
    t = dict(TYPE_PT)
    t.update(type_pt or {})
    ink = pal.neutral.get("ink", "#2B2B2B")
    return {
        "figure.dpi": 100,
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.01,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
        # Use the ASCII hyphen instead of U+2212: it cannot turn into a
        # missing-glyph box if the resolved font's encoding lacks the true minus
        # sign, and a wrong-looking negative tick label is a wrong figure, not
        # merely an ugly one.
        "axes.unicode_minus": False,
        "font.size": t["tick"],
        "axes.titlesize": t["axis_title"],
        "axes.labelsize": t["axis_title"],
        "xtick.labelsize": t["tick"],
        "ytick.labelsize": t["tick"],
        "legend.fontsize": t["legend"],
        "axes.edgecolor": ink,
        "axes.labelcolor": ink,
        "text.color": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": pal.neutral.get("grid", "#E9E9E9"),
        "grid.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "axes.prop_cycle": _mpl_cycler(pal.order),
        "svg.fonttype": "none",  # keep text as text so the SVG stays editable
        "pdf.fonttype": 42,
    }


def _mpl_cycler(colors: Sequence[str]):
    from cycler import cycler

    return cycler(color=list(colors))


def save_mpl(
    fig: Any,
    outdir: str | Path,
    name: str,
    *,
    width_mm: float,
    height_mm: float | None = None,
    dpi: float = DEFAULT_DPI,
    formats: Sequence[str] = ("svg", "png"),
    aspect: float = 0.62,
    relayout: bool = True,
    pad: float = 0.8,
    data: Any | Mapping[str, Any] | None,
) -> dict[str, Path]:
    """:func:`save` for a matplotlib figure — same physical sizing contract.

    Sets the canvas to the requested millimetres and writes without a tight
    bounding box, so the panel really is ``width_mm`` wide and composes with
    plotly panels at matching type size.

    The axes are laid out *after* resizing. Order matters: ``tight_layout``
    stores the axes rectangle in figure fractions, so running it before the
    resize leaves the axis labels hanging off a smaller canvas — where, with no
    tight bounding box to rescue them, they are simply clipped away. Pass
    ``relayout=False`` if you have positioned the axes by hand.

    ``pad`` is in units of font size. Below ~0.6 a rotated y-axis label starts
    to hang off the left edge of a small panel, so trim it only if you check the
    result.

    ``data`` is **required**, exactly as in :func:`save` — see that
    docstring, including the ``NO_BACKING_TABLE`` sentinel for the genuine
    no-table case.
    """
    height_mm = height_mm if height_mm is not None else width_mm * aspect
    fig.set_size_inches(width_mm / MM_PER_IN, height_mm / MM_PER_IN)
    if relayout and type(fig.get_layout_engine()).__name__ != "ConstrainedLayoutEngine":
        fig.tight_layout(pad=pad)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi if fmt == "png" else None, bbox_inches=None)
        written[fmt] = path

    if data is NO_BACKING_TABLE or data is None:
        print(f"[sciplot] {name}: no CSV backup — data={'NO_BACKING_TABLE' if data is NO_BACKING_TABLE else 'None'} "
              f"was passed explicitly to save_mpl(). Only correct for a schematic/diagram or a table "
              f"already written by a separate save_data() call.")
    else:
        written.update(save_data(outdir, name, data))

    print(
        f"[sciplot] {name} (matplotlib): {width_mm:.0f}x{height_mm:.0f} mm -> "
        + ", ".join(str(p) for p in written.values())
    )
    return written
