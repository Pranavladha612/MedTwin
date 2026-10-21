"""Charts for the Canva decks, drawn from the report data (five-seed means).

Two charts, in two colour themes:
  - detectability on heart disease for the four methods
  - share of heart disease patients with each test missing, by hospital

"mint" matches the first deck (white background); "navy" matches the
navy-and-sky-blue deck (transparent background, for the beige paper).

    venv/Scripts/python.exe docs/presentation/make_slide_chart.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "..", "muj_report", "data.json")))

THEMES = {
    "mint": dict(ink="#1c2120", muted="#5b6463", light="#aad7d4", strong="#2a8c85", grid="#e3e8e7",
                 ramp=["#f3faf9", "#aad7d4", "#2a8c85", "#1d5f5a"], bg="white", suffix=""),
    "navy": dict(ink="#1B2A4A", muted="#4a5670", light="#9CC3E6", strong="#1B2A4A", grid="#d6d0c4",
                 ramp=["#eef4fb", "#9CC3E6", "#3f6aa6", "#1B2A4A"], bg="none", suffix="_navy"),
}

models = [("MedTwin WGAN-GP", "MedTwin"), ("TabDDPM", "TabDDPM"), ("TVAE", "TVAE"), ("CTGAN", "CTGAN")]
vals = [DATA["heart"]["models"][k]["detectability_pct"]["mean"] for k, _ in models]
sites = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
tests = [("chol", "Cholesterol"), ("ca", "Vessel scan"), ("trestbps", "Blood pressure")]
mbs = DATA["heart"]["missing_by_site"]
grid = [[mbs[s][c] for c, _ in tests] for s in sites]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20})


def detectability(t):
    out = os.path.join(HERE, f"detectability_heart{t['suffix']}.png")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    width = 0.62
    for i, ((_, name), v) in enumerate(zip(models, vals)):
        color = t["strong"] if i == 0 else t["light"]
        ax.add_patch(FancyBboxPatch((i - width / 2, 0), width, v, boxstyle="round,pad=0,rounding_size=0.06",
                                    mutation_aspect=40, linewidth=0, facecolor=color))
        ax.add_patch(plt.Rectangle((i - width / 2, 0), width, min(v, 4), linewidth=0, facecolor=color))
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", color=t["ink"],
                fontsize=24, fontweight="bold" if i == 0 else "normal")
    ax.axhline(50, color=t["ink"], linewidth=2, linestyle=(0, (6, 4)))
    ax.text(3.45, 50 + 1.2, "50% = can't tell fake from real", ha="right", va="bottom", color=t["muted"], fontsize=15)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([n for _, n in models], color=t["ink"], fontsize=20)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]], color=t["muted"], fontsize=16)
    ax.yaxis.grid(True, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["muted"])
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_title("How often a classifier spots the fakes\n(heart disease, lower is better)", color=t["ink"],
                 fontsize=20, pad=18, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=t["bg"], transparent=t["bg"] == "none")
    plt.close(fig)
    print("wrote", out, [round(v, 1) for v in vals])


def missing_grid(t):
    out = os.path.join(HERE, f"missing_by_hospital{t['suffix']}.png")
    ramp = LinearSegmentedColormap.from_list("ramp", t["ramp"])
    gap = "white" if t["bg"] == "white" else "#efe6d6"
    fig, ax = plt.subplots(figsize=(9, 7.4), dpi=150)
    fig.patch.set_facecolor(t["bg"])
    ax.imshow(grid, cmap=ramp, vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=26,
                    color="white" if v >= 60 else t["ink"], fontweight="bold" if v >= 60 else "normal")
    for k in range(len(tests) + 1):
        ax.axvline(k - 0.5, color=gap, linewidth=6)
    for k in range(len(sites) + 1):
        ax.axhline(k - 0.5, color=gap, linewidth=6)
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels([x for _, x in tests], color=t["ink"], fontsize=20)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, color=t["ink"], fontsize=20)
    ax.tick_params(length=0, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Share of patients with the test missing (real data)", color=t["muted"], fontsize=17, pad=52, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=t["bg"], transparent=t["bg"] == "none")
    plt.close(fig)
    print("wrote", out)


for theme in THEMES.values():
    detectability(theme)
    missing_grid(theme)
