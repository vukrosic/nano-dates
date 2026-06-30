"""Print-friendly (white background) figures for the nano-dates technical-report PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys

# print palette: white paper, dark ink, two muted accents (+ green/amber/red for tiers)
INK = "#1f1e1d"; MUTE = "#6b6864"; PAPER = "#ffffff"
GREEN = "#16a34a"; AMBER = "#d97706"; RED = "#dc2626"; BLUE = "#2563eb"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "text.color": INK, "axes.edgecolor": MUTE,
                     "xtick.color": INK, "ytick.color": INK})

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---- Figure 1: accuracy by category (light) ---------------------------------
data = [
    ("Absolute dates: ISO / long / abbr / D-Mon-Y / ordinal", 100, "parse"),
    ("today / tomorrow / yesterday", 100, "simple"),
    ("next week / last week / next month", 99, "simple"),
    ("in N months", 100, "simple"),
    ("in N weeks", 81, "varN"),
    ("in N days", 78, "varN"),
    ("N days ago", 77, "varN"),
    ("next <weekday>", 13, "weekday"),
    ("last <weekday>", 11, "weekday"),
]
tier_color = {"parse": GREEN, "simple": GREEN, "varN": AMBER, "weekday": RED}
labels = [d[0] for d in data][::-1]
vals = [d[1] for d in data][::-1]
colors = [tier_color[d[2]] for d in data][::-1]

fig, ax = plt.subplots(figsize=(10, 4.9))
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
bars = ax.barh(labels, vals, color=colors, height=0.66, edgecolor=PAPER)
for b, v in zip(bars, vals):
    ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v}%", va="center", ha="left",
            color=INK, fontsize=11.5, fontweight="bold")
ax.set_xlim(0, 110)
ax.set_xlabel("held-out exact-match accuracy (%)", color=MUTE, fontsize=11)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0, labelsize=11)
ax.axvline(100, color=MUTE, lw=0.8, ls=":", alpha=0.6)
legend = [mp.Patch(color=GREEN, label="solved (98-100%)"),
          mp.Patch(color=AMBER, label="partial (77-81%)"),
          mp.Patch(color=RED, label="capacity ceiling (~12%)")]
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
plt.tight_layout()
fig.savefig(f"{OUT}/fig_accuracy_print.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
print("wrote fig_accuracy_print.png")

# ---- Figure 2: the data-generation recipe (light) ---------------------------
fig, ax = plt.subplots(figsize=(10, 3.2))
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

def box(x, y, w, h, text, fc, ec, tc=INK, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                 fc=fc, ec=ec, lw=1.2, mutation_aspect=0.55))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                 color=MUTE, lw=1.5, shrinkA=2, shrinkB=2))

box(2, 10, 20, 10, "1. Sample\nthe ANSWER\nfirst", "#eef4ff", BLUE, BLUE, 11.5, True)
box(29, 10, 23, 10, "2. Render it\nmany natural ways\n(17 surface forms)", "#f4f3f0", MUTE, INK, 10.5)
box(60, 16, 37, 6.2, '"June 12, 2023"   "next friday"   "in 3 weeks"', "#fff7ed", AMBER, "#9a3412", 9.5)
box(60, 6, 37, 6.2, "label = the date you started from\n(correct by construction)", "#ecfdf3", GREEN, "#166534", 9)
arrow(22, 15, 29, 15)
arrow(52, 15, 60, 19)
arrow(52, 15, 60, 9)
plt.tight_layout()
fig.savefig(f"{OUT}/fig_recipe_print.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
print("wrote fig_recipe_print.png")
