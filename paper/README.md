# MedTwin paper (Springer LNCS)

- `main.tex` — the paper, in Springer's LNCS format (`llncs` class).
- `references.bib` — bibliography (39 entries, all cited). Every entry was
  checked against its publisher, Crossref, DataCite or Europe PMC record.
- `tables/` — generated from the saved results; never edit by hand:
  `python research/paper_tables.py`
- `figures/` — copies of `../figures/heart_disease/fig_benchmark.png` and
  `fig_missingness_by_site.png` (made by `python research/visualizer.py --benchmark`).
- `related_work.md` — the earlier related-work draft; `main.tex` now holds
  the current version.

## Building

No LaTeX is installed on this machine. Easiest: on Overleaf, create a project
from the "Springer Lecture Notes in Computer Science" template (it provides
`llncs.cls` and `splncs04.bst`), then upload `main.tex`, `references.bib` and
the `tables/` and `figures/` folders, keeping their paths.

Locally, with `llncs.cls` and `splncs04.bst` next to `main.tex`:

    pdflatex main && bibtex main && pdflatex main && pdflatex main
