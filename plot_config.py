"""
Shared plot style configuration for manuscript figures (targeting eLife).

Usage in any script under analysis/manuscript_example_figs/:

    from plot_config import apply_style, save_fig, get_panel_width, COLORS
    apply_style()

    width = get_panel_width(n_columns=2)
    fig, ax = plt.subplots(figsize=(width, 3))
    # ... plotting ...
    save_fig(fig, "panels/my_panel.png")

eLife requirements reflected here:
- 900 dpi resolution (eLife minimum: 300 dpi)
- RGB color space
- Arial font (sans-serif)
- Full-page width: 20 cm (7.87 in)
- Colour Universal Design for accessibility
- TIFF (8-bit) for final submission; PNG for drafts
"""

import io
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Font sizes (in points)
# ---------------------------------------------------------------------------
FONT_SIZES = {
    "axes_label": 8,
    "tick_label": 7,
    "legend": 7,
    "panel_label": 10,  # A, B, C, ...
    "annotation": 7,
    "colorbar_label": 7,
}

# ---------------------------------------------------------------------------
# Line widths (in points)
# ---------------------------------------------------------------------------
LINE_WIDTHS = {
    "data_trace": 1.0,
    "mean_trace": 2.0,
    "axis_spine": 1.0,
    "tick": 1.0,
    "raster_marker": 0.5,
}

# ---------------------------------------------------------------------------
# Figure sizing (in inches)
# ---------------------------------------------------------------------------
FULL_PAGE_WIDTH = 7.87  # 20 cm — eLife full-page width

FIG_DEFAULTS = {
    "full_page_width": FULL_PAGE_WIDTH,
    "dpi": 900,
}

# ---------------------------------------------------------------------------
# Color palette — discrete colors for conditions / groups
#
# Chosen to be distinguishable under the three common forms of color vision
# deficiency (protanopia, deuteranopia, tritanopia) following Colour Universal
# Design principles. Blue-to-red diverging scheme from ColorBrewer RdBu.
# ---------------------------------------------------------------------------
COLORS = {
    # Diazepam dose-response
    "D0": "#2166AC",
    "D3": "#67A9CF",
    "D10": "#BDBDBD",
    "D30": "#F4A582",
    "D50": "#B2182B",
    # Generic sequential palette for other uses
    "palette": ["#2166AC", "#67A9CF", "#BDBDBD", "#F4A582", "#B2182B"],
}

# Ordered list of conditions for consistent plotting order
CONDITIONS = ["D0", "D3", "D10", "D30", "D50"]

# ---------------------------------------------------------------------------
# parula
#
# Not shipped with matplotlib, so it is defined here from its published 64-entry
# RGB table and registered under the name "parula". Added for the Figure 2
# benchmark heatmaps: those carry a bounded 0-13 score whose top end matters
# most, and "hot" renders that top end white — invisible against the page.
# parula rises monotonically in luminance from dark blue to yellow, so the whole
# range stays readable.
#
# Note that parula originates with MathWorks. Its RGB values are widely
# reproduced, but if the provenance is awkward for a published figure, "viridis"
# is the open, CUD-designed equivalent and is a drop-in replacement here.
# ---------------------------------------------------------------------------
_PARULA_DATA = [
    [0.2422, 0.1504, 0.6603], [0.2504, 0.1650, 0.7076],
    [0.2578, 0.1818, 0.7511], [0.2647, 0.1978, 0.7952],
    [0.2706, 0.2147, 0.8364], [0.2751, 0.2342, 0.8710],
    [0.2783, 0.2559, 0.8991], [0.2803, 0.2782, 0.9221],
    [0.2813, 0.3006, 0.9414], [0.2810, 0.3228, 0.9579],
    [0.2795, 0.3447, 0.9717], [0.2760, 0.3667, 0.9829],
    [0.2699, 0.3892, 0.9906], [0.2602, 0.4123, 0.9952],
    [0.2440, 0.4358, 0.9988], [0.2206, 0.4593, 0.9973],
    [0.1963, 0.4821, 0.9911], [0.1834, 0.5041, 0.9807],
    [0.1786, 0.5250, 0.9678], [0.1764, 0.5450, 0.9532],
    [0.1687, 0.5646, 0.9399], [0.1540, 0.5836, 0.9282],
    [0.1460, 0.6020, 0.9102], [0.1380, 0.6194, 0.8892],
    [0.1248, 0.6363, 0.8664], [0.1113, 0.6529, 0.8420],
    [0.0952, 0.6689, 0.8167], [0.0689, 0.6838, 0.7916],
    [0.0297, 0.6975, 0.7676], [0.0036, 0.7097, 0.7452],
    [0.0067, 0.7203, 0.7243], [0.0433, 0.7295, 0.7047],
    [0.0964, 0.7375, 0.6857], [0.1408, 0.7445, 0.6665],
    [0.1717, 0.7509, 0.6462], [0.1938, 0.7569, 0.6242],
    [0.2161, 0.7627, 0.5999], [0.2470, 0.7681, 0.5731],
    [0.2906, 0.7728, 0.5434], [0.3406, 0.7764, 0.5109],
    [0.3909, 0.7788, 0.4765], [0.4456, 0.7796, 0.4408],
    [0.5044, 0.7788, 0.4040], [0.5616, 0.7770, 0.3684],
    [0.6174, 0.7736, 0.3343], [0.6720, 0.7686, 0.3025],
    [0.7242, 0.7628, 0.2744], [0.7738, 0.7563, 0.2509],
    [0.8203, 0.7498, 0.2326], [0.8634, 0.7441, 0.2190],
    [0.9035, 0.7394, 0.2100], [0.9393, 0.7365, 0.2049],
    [0.9728, 0.7359, 0.2027], [0.9956, 0.7481, 0.1953],
    [0.9970, 0.7719, 0.1774], [0.9952, 0.7961, 0.1585],
    [0.9892, 0.8213, 0.1392], [0.9786, 0.8469, 0.1214],
    [0.9676, 0.8722, 0.1072], [0.9610, 0.8973, 0.0949],
    [0.9597, 0.9221, 0.0836], [0.9628, 0.9461, 0.0747],
    [0.9691, 0.9695, 0.0698], [0.9769, 0.9839, 0.0805],
]

PARULA = mpl.colors.LinearSegmentedColormap.from_list(
    "parula", _PARULA_DATA, N=256
)
if "parula" not in mpl.colormaps:
    mpl.colormaps.register(PARULA, name="parula")


# ---------------------------------------------------------------------------
# Default colormaps
# ---------------------------------------------------------------------------
CMAPS = {
    "heatmap": "hot",
    "diverging": "RdBu_r",
    "score": "parula",  # bounded scores — see the parula note above
}

# ---------------------------------------------------------------------------
# Categorical hues
#
# The palette above is a diverging blue-to-red ramp: it orders conditions, and
# cannot supply several unrelated categories at once. Where a figure needs
# distinct hues for unordered groups, take them from the Okabe-Ito set below,
# which is the reference Colour Universal Design palette and is safe under all
# three common forms of colour vision deficiency.
# ---------------------------------------------------------------------------
CUD_HUES = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


def tint(color, amount):
    """Shift *color* toward white or black, keeping its hue.

    ``amount`` runs -1 to 1: positive lightens toward white, negative darkens
    toward black. Used to separate versions within one model family while
    keeping the family readable as a single hue. Spanning both directions
    roughly doubles the separation available, which matters where a family has
    five generations to tell apart.
    """
    rgb = mpl.colors.to_rgb(color)
    if amount >= 0:
        return tuple(c + (1.0 - c) * amount for c in rgb)
    return tuple(c * (1.0 + amount) for c in rgb)


# ---------------------------------------------------------------------------
# get_panel_width — compute width based on number of columns in the figure
# ---------------------------------------------------------------------------
def get_panel_width(n_columns=1):
    """Return panel width in inches for a panel spanning 1 of *n_columns*.

    Examples:
        get_panel_width(1)  → 7.87  (full page)
        get_panel_width(2)  → 3.94  (half page)
        get_panel_width(3)  → 2.62  (third page)
    """
    return FULL_PAGE_WIDTH / n_columns


# ---------------------------------------------------------------------------
# apply_style — call once at the top of every plotting script
# ---------------------------------------------------------------------------
def apply_style():
    """Apply manuscript figure style to matplotlib rcParams."""
    mpl.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_SIZES["axes_label"],
            # Axes
            "axes.labelsize": FONT_SIZES["axes_label"],
            "axes.titlesize": FONT_SIZES["axes_label"],
            "axes.linewidth": LINE_WIDTHS["axis_spine"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Ticks
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "xtick.major.width": LINE_WIDTHS["tick"],
            "ytick.major.width": LINE_WIDTHS["tick"],
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend
            "legend.fontsize": FONT_SIZES["legend"],
            "legend.frameon": False,
            # Figure
            "figure.dpi": FIG_DEFAULTS["dpi"],
            "savefig.dpi": FIG_DEFAULTS["dpi"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Lines
            "lines.linewidth": LINE_WIDTHS["data_trace"],
            # Image
            "image.cmap": CMAPS["heatmap"],
        }
    )


# ---------------------------------------------------------------------------
# Helper: add panel label (A, B, C, …) to an axes
# ---------------------------------------------------------------------------
def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add a bold uppercase panel label to the top-left corner of an axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FONT_SIZES["panel_label"],
        fontweight="bold",
        va="bottom",
        ha="right",
    )


# ---------------------------------------------------------------------------
# save_fig — save figure as PNG (for panel previews and drafts)
# ---------------------------------------------------------------------------
def save_fig(fig, path):
    """Save figure as PNG at 300 dpi and close it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_fig_submission — save figure as 8-bit RGB TIFF (final assembly)
# ---------------------------------------------------------------------------
def save_fig_submission(fig, path):
    """Save figure as 8-bit RGB TIFF at 300 dpi for eLife submission.

    Used by assembly scripts to render the final composite figure.
    Renders to PNG in memory first, then converts to 8-bit RGB TIFF.
    """
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dpi = FIG_DEFAULTS["dpi"]

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.save(path, format="TIFF", dpi=(dpi, dpi))
