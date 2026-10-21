# MedTwin presentation: slide content (draft 1)

Template: "Blue Minimalist Project Presentation" (Canva). We use only its look: mint/teal accent, bold black headings, small pill labels, rounded mint cards, the donut chart and big-number stat blocks. Layout and content below follow our project.

Plan: 16 slides, about 12–15 minutes. Speaker notes are short; each of the four of us can take a block (suggested split at the end).

All numbers are means over 5 seeds, measured on held-out patients, and match the paper and the MUJ report.

---

## Slide 1: Title

**On slide**
- MedTwin
- Realistic and private synthetic patient data, including the values that were never measured
- Pranav Ladha (23FE10CDS00455) · Devisha Jain (23FE10CDS00493) · Nandini Nahata (23FE10CDS00376) · Jiya Gakhar (23FE10CDS00480)
- Guide: Dr. Chirag Joshi, Assistant Professor (Selection Grade)
- Department of Data Science and Engineering, Manipal University Jaipur

**Visual**: template title slide; MUJ logo top corner; pill label under the title with "B.Tech Project 2026".

**Say**: "Our project is MedTwin. It makes fake patient records that look and behave like real ones, and it keeps the gaps that real hospital data has."

---

## Slide 2: Why medical data is hard to get

**On slide**
- Medical AI needs a lot of real patient records
- Those records are private (HIPAA, GDPR, hospital rules)
- Students and small teams often cannot get access
- Synthetic data: records made by a model that learned from real patients, belonging to no real person

**Visual**: split layout like template slide 2. Left half: a simple mint illustration of a locked patient file. Right half: the four lines. Pill: "The problem".

**Say**: If synthetic data is good enough, it can be shared in place of the real data, for research, teaching and testing software.

---

## Slide 3: What "good" synthetic data means

**On slide** (three rounded cards, like template slide 3)
- Realistic: looks like real data, column by column and in how columns relate
- Useful: a model trained on it works on real patients
- Private: you cannot tell which real patients were used

**Visual**: three mint cards with icons (eye, chart, lock).

**Say**: Most papers check these three. We found a fourth question nobody was checking properly, which is the next slide.

---

## Slide 4: The twist: real data has holes

**On slide**
- Whether a test is done depends on the hospital
- Switzerland: cholesterol missing for 100% of patients
- Vessel scan: missing for 96–99% at three of four hospitals
- VA Long Beach: blood pressure missing for 28%, almost nobody elsewhere
- Usual fix: fill gaps with the median, so fake data has no gaps and is easy to spot

**Visual**: big-number blocks (template slide 4 style) for 100%, 96–99%, 28%, each with a pill naming the hospital.

**Say**: A realistic copy of this data has to reproduce *which* tests were done *where*. Filling the gaps throws that away.

---

## Slide 5: Our questions and objectives

**On slide**
1. Can we generate realistic, useful patients *with* their gaps, hospital by hospital?
2. How do we measure whether the gaps in fake data are realistic?
3. Does keeping realistic gaps create a privacy risk?

Objectives: build the generator, test it fairly against three known methods on two datasets, and create new scores for gaps and their privacy.

**Visual**: three numbered mint circles down the left, objective text on the right.

---

## Slide 6: The data

**On slide**
| | Heart disease (UCI) | Pima diabetes |
|---|---|---|
| Patients | 920 | 768 |
| Hospitals | 4: Cleveland 304, Hungary 293, VA Long Beach 200, Switzerland 123 | 1 clinic |
| Columns | 7 numbers + 7 categories + hospital | 9 numbers |
| Target | heart disease (55.3% yes) | diabetes (34.9% yes) |
| Gaps | many, depend on hospital | zeros that mean "not measured" (insulin 48.7%) |

**Visual**: donut chart (template slide 4) showing the four hospitals' patient counts, table on the side.

---

## Slide 7: How MedTwin works

**On slide** (8-step pipeline)
Settings file → split (80/20) → encode with flags → train → generate 2000 patients → decode → clean → evaluate

- A yes/no "was this measured?" flag for every column that can be missing
- A flag for values that are exactly zero (40% of ST depression values)
- Every patient is generated *for a chosen hospital*

**Visual**: our pipeline block diagram (docs/muj_report/assets/block_pipeline.png), redrawn in mint and black to match the template.

**Say**: The model learns the flags together with the values, so it learns which values tend to be missing and where.

---

## Slide 8: The model

**On slide**
- Conditional WGAN-GP: a generator against a critic, both told the hospital
- Generator: noise (128) + hospital → 256 → 512 → output
- Numbers and flags through sigmoid; categories through Gumbel-softmax (one exact choice)
- Critic: record + hospital → 512 → 256 → realness score
- Loss: Wasserstein + gradient penalty (λ = 10) + correlation loss (α = 2)
- 600 epochs, batch 64, about 7 minutes per run on a laptop CPU

**Visual**: architecture diagram (block_architecture.png) restyled in template colours.

---

## Slide 9: How we tested it

**On slide** (four cards, like template slide 6)
- 20% of patients held out; no model ever sees them
- Every experiment repeated with 5 seeds
- Same data, flags, cleaning and tests for all four methods
- Differences counted only if the paired 95% confidence interval excludes zero

Compared against CTGAN, TVAE and TabDDPM.

**Visual**: four mint cards with icons; small pill row with the three baseline names.

---

## Slide 10: What we measure (two new ideas)

**On slide**
- Realism: fidelity, detectability (50% = can't tell fake from real), correlation gap
- Usefulness: train on fake, test on real (AUC ratio, 1.0 = as good as real)
- Privacy: distance to closest record, membership attack (0.5 = attack fails)
- New: missing-value scores (rate gap, site gap, pattern distance), each compared with two real samples
- New: privacy audit of gaps (attack on patients with rare gap patterns)

**Visual**: two-column layout; the two "New" items in highlighted mint cards.

**Say**: Detectability is a realism score. A model that copied real patients would be impossible to spot and still leak everything, so we never use it as a privacy score.

---

## Slide 11: Results: heart disease

**On slide** (big-number layout around the donut, like template slide 4)
- 61.5%: detectability of MedTwin (others 74.9–92.8%; 50% is ideal)
- 0.975: AUC ratio (tied with TVAE 0.987, well above CTGAN 0.787)
- 87.5%: fidelity, the highest of the four methods
- 0.533: membership attack AUC (0.5 = no leak)

**Visual**: four big numbers with pills (Realism, Realism, Usefulness, Privacy); centre: our benchmark chart (figures/heart_disease/fig_benchmark.png).

**Say**: A classifier can barely pick out our fake patients, and a disease predictor trained on them works almost as well as one trained on real data.

---

## Slide 12: Results: diabetes, and a warning about TabDDPM

**On slide**
- MedTwin again has the best fidelity (90.9%) and ties for usefulness (AUC ratio 0.983)
- TabDDPM is the hardest to detect (55.9%) but it memorised its training patients
- TabDDPM: membership attack AUC 0.803, and 60.8% of its records sit closer to training patients
- MedTwin: AUC 0.494, no sign of copying

**Visual**: side-by-side mint cards "MedTwin" vs "TabDDPM" with the two privacy numbers large.

**Say**: This is why low detectability alone says nothing about privacy.

---

## Slide 13: Are the gaps realistic?

**On slide**
- Pattern distance: MedTwin 0.174, two real samples 0.152
- Other methods: TVAE 0.209, TabDDPM 0.487, CTGAN 0.683
- Site gap: MedTwin 0.022, real 0.024; others 0.035–0.131
- Only MedTwin puts the gaps at the right hospitals

**Visual**: our per-hospital missingness chart (figures/heart_disease/fig_missingness_by_site.png), with the numbers in a mint side panel.

**Say**: Our gaps are almost as close to the real data as a second group of real patients. The other methods got the same flags, but nothing tells them which hospital a gap belongs to.

---

## Slide 14: Does keeping gaps cost privacy?

**On slide**
- About 64 of 368 attacked patients have a rare gap pattern
- Attack on them: AUC 0.575 with gaps, 0.531 without, higher in all 5 seeds
- Common patterns and all patients: no clear difference
- No copying of patterns: unique patterns do not reappear
- Small cost, but on the patients a data owner worries about most

**Visual**: one large number pair "0.531 → 0.575" in the centre, notes around it.

**Say**: This is a new finding. Merging rare patterns before release is the first fix we would test.

---

## Slide 15: What each part adds (ablation)

**On slide** (heart disease)
| Remove | Detectability | Change |
|---|---|---|
| nothing (full model) | 62.2% | |
| missing-value flags | 84.2% | much easier to spot |
| zero flag | 74.9% | easier to spot |
| hospital conditioning | 65.6% | easier to spot, gaps at wrong sites |
| correlation loss | 65.2% | correlation gap 0.041 → 0.075 |

- Training 1200 epochs instead of 600 starts memorising (DCR 53.2% → 55.3%)

**Visual**: horizontal bar chart of the five detectability values in mint, full model highlighted.

---

## Slide 16: Conclusion and future work

**On slide**
Conclusion
- Most realistic records of the four methods, just as useful, no memorisation
- New scores show realistic gaps, per hospital, close to real data
- First evidence that realistic gaps cost a little privacy for unusual patients

Limits: small datasets (under 1000 patients), detectability still above 50%, privacy tested by attacks, with no formal guarantee

Next: compare with MissDiff and Wang et al., protect rare patterns, generate for an unseen hospital, larger and time-based records

**Visual**: SWOT-style 2×2 grid from template slide 9, relabelled "What works / Limits / Next steps / Impact", or a simple two-column layout.

---

## Slide 17: Thank you

**On slide**
- Thank you
- Questions?
- Code: github link (to add) · Paper: "Measuring What Is Missing" (Springer LNCS format)

**Visual**: template closing slide.

---

### Suggested speaker split

- Pranav: slides 1–5 (problem and questions)
- Devisha: slides 6–9 (data, model, testing)
- Nandini: slides 10–13 (measures and main results)
- Jiya: slides 14–17 (privacy, ablation, conclusion, questions)

### Things to decide

- Do we show a live demo of the Streamlit dashboard? If yes, add one slide after slide 9 with its screenshot (docs/muj_report/assets/dashboard.png).
- Is there a time limit? 17 slides fits about 12–15 minutes; for 10 minutes, merge slides 2–3 and 7–8.
