"""Nature-style matplotlib settings for the NC v0.3 figures (Nature Portfolio figure specifications; Wong 2011 colour-blind palette).
Widths: single 89 mm, 1.5 column 120 mm, double 183 mm; max height 170 mm. Text 5-7 pt sans-serif; panel labels 8 pt bold lowercase.
Semantic colours are fixed across all figures."""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
W1, W15, W2, HMAX = 89 * MM, 120 * MM, 183 * MM, 170 * MM
REUSE, REDESIGN, RULE, BASE, LIGHT = '#0072B2', '#D55E00', '#009E73', '#8C8C8C', '#D9D9D9'
ORANGE, SKY, PURPLE, YELLOW, INK = '#E69F00', '#56B4E9', '#CC79A7', '#F0E442', '#1A1A1A'
HIST = PURPLE
# clinical panels (same colour in every clinical figure): routine bundle, matched panel, target-aware panel, age and sex alone
ROUTINE, MATCHED, AWARE, AGESEX = INK, ORANGE, REDESIGN, BASE
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6, 'axes.titlesize': 6.5, 'axes.labelsize': 6, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
    'legend.fontsize': 5.5, 'legend.frameon': False, 'legend.handlelength': 1.2,
    'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.color': INK, 'ytick.color': INK, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 0.8, 'lines.markersize': 3, 'patch.linewidth': 0.5, 'errorbar.capsize': 0, 'axes.grid': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600, 'figure.dpi': 200,
    'axes.titlelocation': 'left', 'axes.titlepad': 3})
DASH = (0, (3, 2))


def label(ax, s, dx=-0.16, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')


def flabel(fig, s, x, y):
    fig.text(x, y, s, fontsize=8, fontweight='bold', va='top', ha='left')


def save(fig, path):
    fig.savefig(str(path) + '.pdf'); fig.savefig(str(path) + '.png', dpi=300); plt.close(fig)
