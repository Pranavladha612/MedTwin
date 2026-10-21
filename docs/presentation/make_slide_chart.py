"""Square detectability chart for the results slide of the Canva deck.

Uses the deck's colours (template mint and near-black, with MedTwin in a
darker teal) and the five-seed heart disease means from the report data.

    venv/Scripts/python.exe docs/presentation/make_slide_chart.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "..", "muj_report", "data.json")))
OUT = os.path.join(HERE, "detectability_heart.png")

INK, MUTED, MINT, TEAL, GRID = "#1c2120", "#5b6463", "#aad7d4", "#2a8c85", "#e3e8e7"
models = [("MedTwin WGAN-GP", "MedTwin"), ("TabDDPM", "TabDDPM"), ("TVAE", "TVAE"), ("CTGAN", "CTGAN")]
vals = [DATA["heart"]["models"][k]["detectability_pct"]["mean"] for k, _ in models]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20})
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

width = 0.62
for i, ((_, name), v) in enumerate(zip(models, vals)):
    color = TEAL if i == 0 else MINT
    # bar from 0 to v with rounded top (4px-ish) anchored square at the baseline
    ax.add_patch(FancyBboxPatch((i - width / 2, 0), width, v, boxstyle="round,pad=0,rounding_size=0.06",
                                mutation_aspect=40, linewidth=0, facecolor=color))
    ax.add_patch(plt.Rectangle((i - width / 2, 0), width, min(v, 4), linewidth=0, facecolor=color))
    ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", color=INK,
            fontsize=24, fontweight="bold" if i == 0 else "normal")

ax.axhline(50, color=INK, linewidth=2, linestyle=(0, (6, 4)))
ax.text(3.45, 50 + 1.2, "50% = can't tell fake from real", ha="right", va="bottom", color=MUTED, fontsize=15)

ax.set_xlim(-0.6, 3.6)
ax.set_ylim(0, 100)
ax.set_xticks(range(len(models)))
ax.set_xticklabels([n for _, n in models], color=INK, fontsize=20)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]], color=MUTED, fontsize=16)
ax.yaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="both", length=0, pad=8)
ax.set_title("How often a classifier spots the fakes\n(heart disease, lower is better)", color=INK,
             fontsize=20, pad=18, loc="left")

fig.tight_layout()
fig.savefig(OUT, facecolor="white")
print("wrote", OUT, [round(v, 1) for v in vals])
plt.close(fig)

# --- Slide 3: share of patients with each test missing, by hospital (real data)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT2 = os.path.join(HERE, "missing_by_hospital.png")
sites = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
tests = [("chol", "Cholesterol"), ("ca", "Vessel scan"), ("trestbps", "Blood pressure")]
mbs = DATA["heart"]["missing_by_site"]
grid = [[mbs[s][c] for c, _ in tests] for s in sites]
ramp = LinearSegmentedColormap.from_list("mint", ["#f3faf9", MINT, TEAL, "#1d5f5a"])

fig, ax = plt.subplots(figsize=(9, 7.4), dpi=150)
fig.patch.set_facecolor("white")
ax.imshow(grid, cmap=ramp, vmin=0, vmax=100, aspect="auto")
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=26,
                color="white" if v >= 60 else INK, fontweight="bold" if v >= 60 else "normal")
# 2px white gaps between cells
for k in range(len(tests) + 1):
    ax.axvline(k - 0.5, color="white", linewidth=6)
for k in range(len(sites) + 1):
    ax.axhline(k - 0.5, color="white", linewidth=6)
ax.set_xticks(range(len(tests)))
ax.set_xticklabels([t for _, t in tests], color=INK, fontsize=20)
ax.xaxis.tick_top()
ax.set_yticks(range(len(sites)))
ax.set_yticklabels(sites, color=INK, fontsize=20)
ax.tick_params(length=0, pad=10)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Share of patients with the test missing (real data)", color=MUTED, fontsize=17, pad=52, loc="left")
fig.tight_layout()
fig.savefig(OUT2, facecolor="white")
print("wrote", OUT2)
