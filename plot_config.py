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
# Default colormaps
# ---------------------------------------------------------------------------
CMAPS = {
    "heatmap": "hot",
    "diverging": "RdBu_r",
}


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
