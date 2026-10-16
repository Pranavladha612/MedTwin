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

`MedTwin_paper.pdf` is the compiled paper (15 pages: about 12 of text and
3 of references). To rebuild it after changes, with Tectonic
(https://tectonic-typesetting.github.io, a single program that downloads
the LaTeX packages it needs):

    cd paper
    tectonic --outdir build main.tex
    copy build\main.pdf MedTwin_paper.pdf

On this computer Tectonic is at `%LOCALAPPDATA%\tectonic\tectonic.exe`.

Overleaf also works: create a project from the "Springer Lecture Notes in
Computer Science" template, then upload `main.tex`, `references.bib` and
the `tables/` and `figures/` folders, keeping their paths.
