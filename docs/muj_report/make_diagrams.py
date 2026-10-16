"""Block diagrams for the MUJ-format report (assets/*.png).

    python docs/muj_report/make_diagrams.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
BLUE, BLUE_BG, ORANGE, ORANGE_BG, GREEN, GREEN_BG = "#1f5fa8", "#e8f0fa", "#c4511f", "#fdf0ea", "#1b7f5a", "#e6f4ee"
INK, GREY = "#1d1d1f", "#55555a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def box(ax, x, y, w, h, title, body="", edge=BLUE, face=BLUE_BG, title_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.6, edgecolor=edge, facecolor=face))
    if body:
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=title_size, weight="bold", color=edge)
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, weight="bold", color=edge)


def arrow(ax, x1, y1, x2, y2, color=GREY, text=None, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 linewidth=1.5, color=color, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=8, color=GREY)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def pipeline():
    fig, ax = canvas(11, 4.6)
    top = [("1. Settings file", "columns, rules,\nzeros = missing"),
           ("2. Train / test split", "80% train\n20% held out"),
           ("3. Encode", "missing flags, zero flag,\nscale, one-hot"),
           ("4. Train MedTwin", "conditional WGAN-GP\n600 epochs")]
    bottom = [("8. Evaluate", "realism, usefulness,\nprivacy, missing values"),
              ("7. Clean", "range check,\nrounding"),
              ("6. Decode", "real units; blank if\nflag = not measured"),
              ("5. Generate", "2000 patients, each\nfor a hospital")]
    w, h, gap = 2.35, 1.2, 0.35
    for i, (t, b) in enumerate(top):
        x = 0.25 + i * (w + gap)
        box(ax, x, 2.95, w, h, t, b)
        if i:
            arrow(ax, x - gap, 2.95 + h / 2, x, 2.95 + h / 2)
    for i, (t, b) in enumerate(bottom):
        x = 0.25 + i * (w + gap)
        edge, face = (GREEN, GREEN_BG) if i == 0 else (BLUE, BLUE_BG)
        box(ax, x, 0.55, w, h, t, b, edge=edge, face=face)
        if i:
            arrow(ax, x, 0.55 + h / 2, x - gap, 0.55 + h / 2)
    x_last = 0.25 + 3 * (w + gap) + w / 2
    arrow(ax, x_last, 2.95, x_last, 0.55 + h)
    ax.text(0.25, 4.35, "Real patient records (CSV)  +  dataset settings (YAML)", fontsize=9.5, color=GREY)
    ax.text(0.25, 0.12, "Output: synthetic patient records, evaluation scores (JSON), report and dashboard",
            fontsize=9.5, color=GREY)
    save(fig, "block_pipeline.png")


def architecture():
    fig, ax = canvas(11, 5.6)
    box(ax, 0.2, 3.6, 1.9, 1.1, "Noise z", "128 random\nnumbers", edge=GREY, face="#f3f3f4")
    box(ax, 0.2, 1.9, 1.9, 1.1, "Hospital c", "one-hot\ncondition", edge=ORANGE, face=ORANGE_BG)
    box(ax, 2.9, 2.55, 2.4, 1.75, "Generator G", "Linear 256 + LeakyReLU\n+ BatchNorm\nLinear 512 + LeakyReLU\n+ BatchNorm\nLinear (output)")
    box(ax, 6.05, 3.55, 2.1, 1.25, "Sigmoid outputs", "numbers + missing\nflags + zero flag", edge=BLUE, face="white")
    box(ax, 6.05, 1.95, 2.1, 1.25, "Gumbel-softmax", "one category per\ncategorical column", edge=BLUE, face="white")
    box(ax, 8.85, 2.55, 2.0, 1.75, "Critic D", "Linear 512 + Dropout\nLinear 256 + Dropout\nLinear 1\n(realness score)",
        edge=GREEN, face=GREEN_BG)
    box(ax, 8.85, 0.3, 2.0, 1.25, "Real patients", "same hospital\nas condition", edge=GREY, face="#f3f3f4")
    arrow(ax, 2.1, 4.15, 2.9, 3.75)
    arrow(ax, 2.1, 2.45, 2.9, 3.0)
    arrow(ax, 5.3, 3.6, 6.05, 4.15)
    arrow(ax, 5.3, 3.25, 6.05, 2.6)
    arrow(ax, 8.15, 4.15, 8.85, 3.75)
    arrow(ax, 8.15, 2.6, 8.85, 3.05)
    arrow(ax, 9.85, 1.55, 9.85, 2.55)
    ax.add_patch(FancyArrowPatch((1.15, 1.9), (9.0, 2.55), connectionstyle="arc3,rad=0.28", arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.3, color=ORANGE, linestyle="--"))
    ax.text(4.6, 0.75, "hospital also given to the critic", fontsize=8.5, color=ORANGE)
    ax.text(0.2, 5.3, "Loss = Wasserstein loss + gradient penalty (λ = 10) + correlation penalty (α = 2)",
            fontsize=9.5, color=INK, weight="bold")
    save(fig, "block_architecture.png")


def evaluation():
    fig, ax = canvas(11, 5.3)
    box(ax, 0.2, 3.2, 2.2, 1.2, "Real data", "920 / 768\npatients", edge=GREY, face="#f3f3f4")
    box(ax, 3.0, 3.75, 2.2, 0.95, "Training set (80%)", edge=BLUE, face=BLUE_BG, title_size=9.5)
    box(ax, 3.0, 2.15, 2.2, 0.95, "Held-out set (20%)", edge=ORANGE, face=ORANGE_BG, title_size=9.5)
    box(ax, 5.9, 3.75, 2.0, 0.95, "Generator", edge=BLUE, face="white", title_size=9.5)
    box(ax, 5.9, 2.15, 2.0, 0.95, "Synthetic data", edge=BLUE, face="white", title_size=9.5)
    tests = [("Realism", "fidelity, detectability,\ncorrelation gap"),
             ("Usefulness", "TSTR vs TRTR AUC"),
             ("Privacy", "DCR, membership\ninference"),
             ("Missing values", "rate, site and pattern\ngaps; rare-pattern attack")]
    for i, (t, b) in enumerate(tests):
        box(ax, 8.6, 4.0 - i * 1.08, 2.3, 1.0, t, b, edge=GREEN, face=GREEN_BG, title_size=9.5)
    arrow(ax, 2.4, 3.95, 3.0, 4.2)
    arrow(ax, 2.4, 3.55, 3.0, 2.6)
    arrow(ax, 5.2, 4.22, 5.9, 4.22)
    arrow(ax, 6.9, 3.75, 6.9, 3.1)
    arrow(ax, 7.9, 2.62, 8.6, 2.62)
    arrow(ax, 5.2, 2.35, 8.6, 1.3)
    ax.text(5.55, 1.45, "never seen by the generator", fontsize=8.5, color=ORANGE, rotation=-15)
    ax.text(0.2, 0.15, "Repeated for 5 seeds (different split and start each time); models compared seed by seed "
                       "with paired 95% confidence intervals.", fontsize=9, color=GREY)
    save(fig, "block_evaluation.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline()
    architecture()
    evaluation()
