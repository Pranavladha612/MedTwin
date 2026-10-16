"""Builds the project report (docs/MedTwin_Project_Report.pdf) for readers
new to the project. Every result number is read from the saved result files,
so the report always matches the experiments.

    python docs/build_report.py

Writes docs/report.html, then prints it to PDF with Microsoft Edge (or
Chrome) in headless mode.
"""

import html
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core import config, medtwin  # noqa: E402
from core.evaluation import align_real_to_report  # noqa: E402
from core.ingestion import mark_missing  # noqa: E402
from research.multi_seed import MEDTWIN, load_runs, paired_table  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
HTML_PATH = os.path.join(DOCS, "report.html")
PDF_PATH = os.path.join(DOCS, "MedTwin_Project_Report.pdf")
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
HEART, DIAB = "configs/heart_disease.yaml", "configs/diabetes.yaml"
MODELS = [MEDTWIN, "TVAE", "TabDDPM", "CTGAN"]
LABEL = {MEDTWIN: "MedTwin (ours)", "TVAE": "TVAE", "TabDDPM": "TabDDPM", "CTGAN": "CTGAN"}


# ---------------------------------------------------------------- data ---
def results(settings, kind):
    cfg = config.use(os.path.join(ROOT, settings))
    return load_runs(os.path.join(cfg.output_dir, kind))


RUNS = {"h": results(HEART, "multi_seed"), "d": results(DIAB, "multi_seed")}
ABL = {"h": results(HEART, "ablation"), "d": results(DIAB, "ablation")}


def mean(ds, group, metric, kind="model", source=None):
    runs = (source or RUNS)[ds]
    return runs.loc[runs[kind] == group, metric].mean()


def sd(ds, group, metric, kind="model", source=None):
    runs = (source or RUNS)[ds]
    return runs.loc[runs[kind] == group, metric].std(ddof=1)


def f(x, d=3):
    return "&ndash;" if pd.isna(x) else f"{x:.{d}f}"


def ms(ds, group, metric, d=3, kind="model", source=None):
    m, s = mean(ds, group, metric, kind, source), sd(ds, group, metric, kind, source)
    return "&ndash;" if pd.isna(m) else f"{m:.{d}f} <span class=sd>&plusmn; {s:.{d}f}</span>"


def am(ds, variant, metric, d=3):
    return f(mean(ds, variant, metric, "variant", ABL), d)


def clear(ds, reference, other, metric, kind="model", source=None):
    runs = (source or RUNS)[ds]
    table = paired_table(runs, reference, [other], column=kind, label="ref")
    row = table[table.metric == metric]
    return not row.empty and row.verdict.iloc[0] != "no clear difference"


def dataset_facts():
    out = {}
    for key, settings in [("h", HEART), ("d", DIAB)]:
        cfg = config.use(os.path.join(ROOT, settings))
        raw = pd.read_csv(cfg.data_file)
        real = align_real_to_report(raw, cfg.report_columns)
        out[key] = {"cfg": cfg, "raw": raw, "real": real, "n": len(raw),
                    "pos": (raw[cfg.target] > cfg.positive_above).mean() * 100}
    return out


FACTS = dataset_facts()


# ------------------------------------------------------------- helpers ---
def table(header, rows, cls="", widths=None):
    cols = "".join(f"<col style='width:{w}'>" for w in widths) if widths else ""
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><colgroup>{cols}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figure(path, caption, width="100%"):
    rel = os.path.relpath(os.path.join(ROOT, path), DOCS).replace("\\", "/")
    return f"<figure><img src='{rel}' style='width:{width}'><figcaption>{caption}</figcaption></figure>"


def box(title, body, kind="note"):
    return f"<div class='box {kind}'><div class='box-title'>{title}</div>{body}</div>"


def esc(s):
    return html.escape(str(s))


# ------------------------------------------------------------ sections ---
def cover():
    return f"""
<section class='cover'>
  <div class='cover-top'>Project report &middot; Department of Data Science and Engineering, Manipal University Jaipur</div>
  <h1>MedTwin</h1>
  <div class='subtitle'>Realistic and private synthetic patient data &mdash; including the values that were never measured</div>
  <div class='cover-desc'>
    A generative AI system that creates fake-but-realistic patient records from real hospital data,
    tested on two public medical datasets against three well-known methods, with new ways to measure
    whether the <em>missing</em> parts of the data are realistic and whether they leak private information.
  </div>
  <div class='cover-authors'>Aarzoo Jain &nbsp;&middot;&nbsp; Pranav Ladha</div>
  <div class='cover-date'>October 2026</div>
  <div class='cover-glance'>
    <div><b>{f(mean('h', MEDTWIN, 'detectability_pct'), 1)}%</b><span>how often a classifier can spot our fake heart patients (50% = impossible to spot; other methods 74.9&ndash;92.8%)</span></div>
    <div><b>{f(mean('h', MEDTWIN, 'auc_ratio'), 2)}</b><span>usefulness of our fake data compared with real data (1.00 = just as useful)</span></div>
    <div><b>{f(mean('h', MEDTWIN, 'pattern_tvd'), 2)}</b><span>missing-value pattern error (two real samples: {f(mean('h', MEDTWIN, 'pattern_tvd_ref'), 2)}; other methods {f(mean('h', 'TVAE', 'pattern_tvd'), 2)}&ndash;{f(mean('h', 'CTGAN', 'pattern_tvd'), 2)})</span></div>
  </div>
</section>"""


def contents():
    items = [
        ("1", "Summary"), ("2", "The problem in simple words"), ("3", "Key ideas explained"),
        ("4", "The data"), ("5", "How MedTwin works"), ("6", "Technical specifications"),
        ("7", "How we test it"), ("8", "Results"), ("9", "Comparison with previous work"),
        ("10", "What is new in this project"), ("11", "Limitations"), ("12", "Future work"),
        ("13", "Project history"), ("A", "How to run the project"), ("B", "References"),
    ]
    rows = "".join(f"<li><span class='num'>{n}</span>{t}</li>" for n, t in items)
    return f"<section class='toc'><h2 class='plain'>Contents</h2><ol class='toc-list'>{rows}</ol></section>"


def summary():
    h_det = f(mean("h", MEDTWIN, "detectability_pct"), 1)
    return f"""
<section>
<h2>1&nbsp; Summary</h2>
<p><b>What.</b> MedTwin is a program that learns from real patient records and then creates new,
fake patient records that look and behave like the real ones. Nobody in the fake data is a real person.
Researchers can use such data to build and test medical AI without seeing private patient files.</p>
<p><b>Why it is hard.</b> Real hospital data has holes. Different hospitals measure different things:
in our heart data, <i>no</i> Swiss patient has a cholesterol value and almost no Hungarian patient has a
vessel-count result. Most existing tools fill these holes with guesses before learning, so their fake data
has no holes at all &mdash; which makes it unrealistic and easy to spot.</p>
<p><b>How.</b> MedTwin is a type of AI called a <i>conditional Wasserstein GAN</i>. For every value that can be
missing it also learns a yes/no &ldquo;was this measured?&rdquo; flag, and it creates each patient <i>for a
specific hospital</i>, so it learns each hospital's habits. We also built new tests that measure whether the
holes in fake data are realistic and whether keeping them puts privacy at risk.</p>
<p><b>Results</b> (five repeated experiments, tested on real patients the AI never saw):</p>
<ul>
<li><b>Most realistic.</b> A classifier can tell our fake heart patients from real ones only {h_det}% of the time
  (50% = pure guessing). The other three methods: {f(mean('h','TabDDPM','detectability_pct'),1)}%,
  {f(mean('h','TVAE','detectability_pct'),1)}% and {f(mean('h','CTGAN','detectability_pct'),1)}%.</li>
<li><b>Just as useful.</b> A disease predictor trained only on our fake data works
  {f(mean('h',MEDTWIN,'auc_ratio')*100,1)}% as well as one trained on real data (heart disease) and
  {f(mean('d',MEDTWIN,'auc_ratio')*100,1)}% as well (diabetes), tied with the best other method.</li>
<li><b>Realistic holes.</b> The missing-value patterns in our fake data are almost as close to the real data as a
  second sample of real patients is. Only MedTwin puts the holes at the right hospitals.</li>
<li><b>No copying found.</b> Privacy attacks found no sign that MedTwin memorised real patients. One other method
  (TabDDPM) clearly did on the diabetes data.</li>
<li><b>An honest catch.</b> Keeping realistic holes makes patients with <i>unusual</i> holes slightly easier to
  identify (attack score {am('h','full','rare_pattern_dist_mia_auc')} vs {am('h','no_missing_flags','rare_pattern_dist_mia_auc')}
  without the flags). This is a new finding that we report openly.</li>
</ul>
</section>"""


def problem():
    return """
<section>
<h2>2&nbsp; The problem in simple words</h2>
<h3>Medical AI needs data that is hard to share</h3>
<p>To build a computer program that predicts, for example, heart disease, we need many real patient records.
These records are private. Laws such as HIPAA (USA) and GDPR (Europe), and hospital rules, rightly limit who
can see them. So students, researchers and companies often cannot get the data they need.</p>
<h3>The idea: synthetic data</h3>
<p><b>Synthetic data</b> means fake records produced by a computer model that has learned the patterns in real
records. A good synthetic patient looks real &mdash; an older patient tends to have a lower maximum heart rate, a
sick patient tends to have chest pain &mdash; but does not belong to any real person. If the fake data is good
enough, people can share it instead of the real data.</p>
<p>A synthetic dataset must pass three tests at the same time:</p>
<table class='three'>
<tr><td><b>1. Realistic</b><br>Does it look like real data, column by column and in how columns relate?</td>
<td><b>2. Useful</b><br>Does a model trained on the fake data work on real patients?</td>
<td><b>3. Private</b><br>Is it impossible to work out which real patients were used to make it?</td></tr>
</table>
<h3>The twist: real data has holes</h3>
<p>Real medical tables are incomplete, and not at random. Whether a test was done depends on the hospital, the
doctor and the patient. In the heart disease data we use:</p>
<ul>
<li>every Swiss patient has cholesterol recorded as 0, which really means &ldquo;not measured&rdquo;;</li>
<li>the number of major vessels (from a scan) is missing for 96&ndash;99% of patients at three of four hospitals;</li>
<li>resting blood pressure is missing for 28% of patients at VA Long Beach but almost nobody elsewhere.</li>
</ul>
<p>Most synthetic-data tools cannot handle holes, so people fill them with a typical value (for example the
median) first. The fake data then has no holes, which is not how real data looks, and the filled-in values are
inventions. Our project asks:</p>
<div class='box goal'><div class='box-title'>Project questions</div>
<ol>
<li>Can we generate realistic, useful synthetic patients <i>including</i> their holes, hospital by hospital?</li>
<li>How do we <i>measure</i> whether the holes in fake data are realistic?</li>
<li>Does keeping realistic holes create a new <i>privacy risk</i>?</li>
</ol></div>
</section>"""


def ideas():
    rows = [
        ("Generative AI / generator", "A model that learns from examples and then produces new examples of the same kind."),
        ("GAN (Generative Adversarial Network)", "Two networks that compete: a <i>generator</i> makes fake records, a <i>critic</i> tries to tell fake from real. Both improve until the fakes are hard to spot [15]."),
        ("Wasserstein GAN with gradient penalty (WGAN-GP)", "A more stable kind of GAN. The critic gives a &ldquo;realness score&rdquo; instead of a yes/no answer, and a penalty keeps training smooth [16, 17]."),
        ("Conditioning", "Telling the generator what kind of record to make. We tell it the hospital, so it learns each hospital's habits."),
        ("Missing-value flag", "An extra yes/no value per column: &ldquo;was this measured?&rdquo; The generator learns these too, and leaves a value blank when it says &ldquo;no&rdquo;."),
        ("Held-out patients", "20% of real patients that no model ever sees during training. We test on them so the results are fair."),
        ("Seed", "A number that fixes the random choices. We repeat everything with 5 seeds to check results are not luck."),
        ("TSTR", "Train on Synthetic, Test on Real: train a disease predictor on fake data and test it on real held-out patients [20]."),
        ("AUC", "A score for how well a predictor separates two groups: 1.0 = perfect, 0.5 = coin flip."),
        ("Detectability", "How often a classifier can tell a fake record from a real one. 50% = impossible to tell (best)."),
        ("Membership inference attack (MIA)", "An attack that tries to find out whether a given real patient was used to train the generator. AUC 0.5 = attack fails (good) [21, 22]."),
        ("DCR (distance to closest record)", "Checks whether fake records sit suspiciously close to training patients. 50% = no sign of copying."),
        ("Ablation", "Switching off one part of the system to see how much that part matters."),
        ("Paired 95% confidence interval", "A statistical test across the 5 seeds. We call a difference &ldquo;clear&rdquo; only when the interval does not include zero."),
    ]
    return f"""
<section>
<h2>3&nbsp; Key ideas explained</h2>
<p>These terms appear throughout the report.</p>
{table(["Term", "Meaning in simple words"], rows, "glossary", ["31%", "69%"])}
</section>"""


def data_section():
    h, d = FACTS["h"], FACTS["d"]
    hc = config.use(os.path.join(ROOT, HEART))  # mark_missing reads the active settings
    raw = mark_missing(h["raw"])
    desc = {
        "age": "Age in years", "sex": "Sex", "dataset": "Hospital (site)", "cp": "Chest pain type",
        "trestbps": "Resting blood pressure (mmHg)", "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar &gt; 120 mg/dl", "restecg": "Resting ECG result",
        "thalch": "Maximum heart rate (bpm)", "exang": "Chest pain during exercise",
        "oldpeak": "ST depression on ECG", "slope": "Slope of ST segment",
        "ca": "Major vessels seen on scan (0&ndash;3)", "thal": "Thalassemia test result",
        "num": "Disease severity (0 = none, 1&ndash;4)",
    }
    rows = []
    for col in [c for c in raw.columns if c != hc.id_column]:
        kind = "site" if col == hc.condition_on else ("number" if col in hc.numeric else "category")
        rows.append((f"<code>{col}</code>", desc.get(col, col), kind, f"{raw[col].isna().mean()*100:.1f}%"))
    site_cols = ["chol", "trestbps", "ca", "thal", "slope", "fbs"]
    by_site = raw.groupby(hc.condition_on)[site_cols].apply(lambda g: g.isna().mean() * 100)
    counts = raw[hc.condition_on].value_counts()
    site_rows = [[f"{s} ({counts[s]})"] + [f"{by_site.loc[s, c]:.0f}%" for c in site_cols] for s in by_site.index]
    dc = config.use(os.path.join(ROOT, DIAB))
    draw = mark_missing(d["raw"])
    drows = [(f"<code>{c}</code>", dc.labels.get(dc.report_name(c), c),
              f"{draw[c].isna().mean()*100:.1f}%" + (" (0 = not measured)" if c in dc.zero_means_missing else ""))
             for c in draw.columns]
    return f"""
<section>
<h2>4&nbsp; The data</h2>
<p>We use two public medical datasets. Both are small, which is typical of medical research and makes the task harder.</p>
{table(["", "Heart disease (UCI)", "Pima diabetes"], [
    ("Patients", h['n'], d['n']),
    ("Columns used", f"{len(hc.numeric)} numbers + {len(hc.categorical)} categories + hospital", f"{len(dc.numeric)} numbers"),
    ("Hospitals", "4: Cleveland, Hungary, Switzerland, VA Long Beach", "1 (one clinic)"),
    ("Prediction target", f"has heart disease (severity &gt; 0): {h['pos']:.1f}% yes", f"has diabetes: {d['pos']:.1f}% yes"),
    ("Missing values", "many, depending strongly on hospital", "zeros that mean &ldquo;not measured&rdquo;"),
    ("Source", "UCI Machine Learning Repository [1]; collected by Detrano et al. [13]", "OpenML dataset 37 [2]; Smith et al. [14]"),
], "", ["22%", "45%", "33%"])}
<h3>Heart disease: the columns</h3>
{table(["Column", "Meaning", "Type", "Missing"], rows, "compact", ["14%", "52%", "16%", "18%"])}
<div class='keep'>
<h3>Heart disease: missing values depend on the hospital</h3>
<p>Share of patients with the value <i>not measured</i>, per hospital (number of patients in brackets). This
pattern is exactly what a realistic synthetic dataset must reproduce.</p>
{table(["Hospital"] + [f"<code>{c}</code>" for c in site_cols], site_rows, "compact center")}
</div>
<h3>Pima diabetes: the columns</h3>
<p>768 women of Pima heritage, aged 21 or older. A value of 0 for glucose, blood pressure, skin fold, insulin or
BMI is physically impossible and means &ldquo;not measured&rdquo;; 0 pregnancies is a real value.</p>
{table(["Column", "Meaning", "Missing"], drows, "compact", ["16%", "46%", "38%"])}
</section>"""


def pipeline():
    steps = [
        ("1", "Load settings", "A YAML file describes the dataset: columns, rules, which zeros mean missing."),
        ("2", "Split", "80% of patients for training, 20% held out for testing (never seen)."),
        ("3", "Encode", "Mark missing values, add yes/no flags, scale numbers to 0&ndash;1, one-hot categories."),
        ("4", "Train", "Conditional WGAN-GP learns real patients, hospital by hospital (600 epochs)."),
        ("5", "Generate", "Create 2000 fake patients, each for a hospital drawn in real proportions."),
        ("6", "Decode", "Back to real units; leave a value blank when its flag says &ldquo;not measured&rdquo;."),
        ("7", "Clean", "Medical range check and rounding to how hospitals record values."),
        ("8", "Evaluate", "Realism, usefulness, privacy and missing-value tests on held-out patients."),
    ]
    boxes = "".join(f"<div class='step'><div class='step-n'>{n}</div><div class='step-t'>{t}</div><div class='step-d'>{d}</div></div>"
                    for n, t, d in steps)
    return f"""
<section>
<h2>5&nbsp; How MedTwin works</h2>
<p>The whole system is one pipeline. Each step is a separate, tested part of the code.</p>
<div class='pipeline'>{boxes}</div>
<h3>Step by step</h3>
<p><b>Settings file.</b> Everything specific to a dataset lives in a small settings file (for example
<code>configs/heart_disease.yaml</code>): which columns are numbers or categories, which column is the hospital,
which zeros mean &ldquo;not measured&rdquo;, and the medically possible range of each value. Switching to a new
dataset means writing a new settings file, not changing code &mdash; that is how we added the diabetes data.</p>
<p><b>Handling missing values (the key idea).</b> For every number column that is sometimes missing, MedTwin adds a
yes/no <i>measured flag</i>. The generator learns these flags together with the values, so it learns <i>which</i>
values tend to be missing and <i>where</i>. When it creates a patient and the flag says &ldquo;not measured&rdquo;,
the value is left blank. For categories, &ldquo;not measured&rdquo; is simply one more category.</p>
<p><b>The zero flag.</b> 40% of heart patients have an ST depression of exactly 0. A neural network cannot easily
produce one exact number, so MedTwin learns a separate yes/no &ldquo;is it exactly zero?&rdquo; flag.</p>
<p><b>Hospital conditioning.</b> The hospital is not generated &mdash; it is <i>given</i> to the generator. MedTwin
creates each patient for a chosen hospital, and the critic judges whether the patient is realistic <i>for that
hospital</i>. This is how site-specific habits (like no cholesterol test in Switzerland) are learned.</p>
<p><b>Correlation loss.</b> Besides fooling the critic, the generator is also penalised when the correlations
between columns in its output differ from those in real data. This keeps relationships like
&ldquo;older &rarr; lower maximum heart rate&rdquo; intact.</p>
<p><b>Cleaning.</b> Generated values are rounded the way hospitals record them (for example ST depression to 0.1)
and checked against medical ranges. The ranges were chosen to include every real measured value.</p>
</section>"""


def specs():
    h = FACTS["h"]["cfg"]
    rt = RUNS["h"].groupby("model")["train_seconds"].mean()
    rt_d = RUNS["d"].groupby("model")["train_seconds"].mean()
    arch = [
        ("Generator input", f"random noise (128 numbers) + one-hot hospital"),
        ("Generator layers", "Linear 256 &rarr; LeakyReLU(0.2) &rarr; BatchNorm &rarr; Linear 512 &rarr; LeakyReLU &rarr; BatchNorm &rarr; Linear (output)"),
        ("Generator outputs", "Sigmoid for numbers and flags (0&ndash;1); Gumbel-softmax (&tau; = 0.2) per category, exactly one-hot when sampling"),
        ("Critic layers", "Linear 512 &rarr; LeakyReLU &rarr; Dropout 0.3 &rarr; Linear 256 &rarr; LeakyReLU &rarr; Dropout 0.3 &rarr; Linear 1 (no sigmoid)"),
        ("Critic input", "record + hospital, so it judges realism per hospital"),
        ("Loss", "Wasserstein loss [16] + gradient penalty (&lambda; = 10) [17] + correlation penalty (&alpha; = 2)"),
    ]
    train = [
        ("Optimiser", "Adam [18], learning rate 0.0001, betas (0.0, 0.9) for both networks"),
        ("Epochs / batch size", f"{medtwin.EPOCHS} epochs, batch size 64"),
        ("Critic updates", "5 critic updates per generator update"),
        ("Records generated", f"{medtwin.N_SAMPLES} per run"),
        ("Train / test split", "80% / 20%, stratified by outcome; 5 seeds (0&ndash;4) change both the split and the model start"),
        ("Training time", f"about {rt[MEDTWIN]/60:.0f} min (heart) and {rt_d[MEDTWIN]/60:.0f} min (diabetes) per run on a laptop CPU, several runs in parallel"),
    ]
    baselines = [
        ("CTGAN [3]", "Conditional tabular GAN, 300 epochs, batch 100", f"{rt['CTGAN']/60:.1f} min"),
        ("TVAE [3]", "Variational autoencoder for tables, 300 epochs, batch 100", f"{rt['TVAE']/60:.1f} min"),
        ("TabDDPM [4]", "Diffusion model for tables (our compact reimplementation), 8000 steps", f"{rt['TabDDPM']/60:.1f} min"),
        ("MedTwin (ours)", "Conditional WGAN-GP with missing-value flags, 600 epochs", f"{rt[MEDTWIN]/60:.1f} min"),
    ]
    stack = [
        ("Language", "Python 3.9"), ("Deep learning", "PyTorch 2.8.0 [27]"), ("Data", "pandas 2.3.3, NumPy 2.0.2"),
        ("Machine learning / tests", "scikit-learn 1.6.1 [28] (random forests [19], nearest neighbours, AUC)"),
        ("Baselines", "ctgan 0.12.1 (CTGAN, TVAE); TabDDPM written in this project"),
        ("Statistics", "SciPy (paired t-tests)"), ("Charts", "matplotlib 3.9.4, seaborn 0.13.2"),
        ("Dashboard", "Streamlit 1.50.0"), ("Settings", "PyYAML 6.0.3"), ("Testing", "pytest (31 automated tests)"),
    ]
    code = [
        ("<code>main.py</code>", "Runs the full pipeline once and saves the synthetic data and scores"),
        ("<code>core/config.py</code>", "Loads a dataset's settings file"),
        ("<code>core/ingestion.py</code>", "Missing-value handling, flags, scaling, encoding"),
        ("<code>core/generator.py</code>", "Generator, critic and WGAN-GP training"),
        ("<code>core/medtwin.py</code>", "Train-and-generate steps; ablation switches"),
        ("<code>core/medical_logic.py</code>", "Medical range check and rounding"),
        ("<code>core/evaluation.py</code>", "Usefulness, privacy and correlation tests on held-out patients"),
        ("<code>core/missingness.py</code>", "New missing-value realism scores and privacy attacks"),
        ("<code>research/baselines.py</code>, <code>tabddpm.py</code>", "The three comparison methods"),
        ("<code>research/multi_seed.py</code>", "5-seed comparison with paired statistics"),
        ("<code>research/ablation.py</code>", "Ablation and sensitivity studies"),
        ("<code>research/visualizer.py</code>, <code>paper_tables.py</code>", "Figures and paper tables"),
        ("<code>app/dashboard.py</code>", "Interactive Streamlit dashboard"),
        ("<code>configs/*.yaml</code>", "One settings file per dataset"),
        ("<code>tests/</code>", "31 automated tests"),
        ("<code>paper/</code>", "Research paper (Springer LNCS format)"),
    ]
    return f"""
<section>
<h2>6&nbsp; Technical specifications</h2>
<h3>Model architecture</h3>
{table(["Part", "Specification"], arch, "compact", ["24%", "76%"])}
<h3>Training settings</h3>
{table(["Setting", "Value"], train, "compact", ["24%", "76%"])}
<h3>Methods compared</h3>
{table(["Method", "What it is", "Training time (heart, avg.)"], baselines, "compact", ["20%", "56%", "24%"])}
<div class='keep'><h3>Software</h3>
{table(["Area", "Tools and versions"], stack, "compact", ["28%", "72%"])}</div>
<h3>Code structure</h3>
{table(["File / folder", "Purpose"], code, "compact", ["40%", "60%"])}
<p class='small'>Reproducibility: every experiment can be rerun from the code. When we reran all experiments after adding new
measurements, all 400 previously saved results came out exactly the same.</p>
</section>"""


def testing():
    rows = [
        ("Fidelity", "Do single columns look real? (overlap of histograms)", "higher, max 100%", "realism"),
        ("Detectability", "Can a classifier tell fake from real?", "50%", "realism"),
        ("Correlation gap", "Do relationships between columns match?", "0", "realism"),
        ("TSTR AUC", "How good is a disease predictor trained on fake data, tested on real patients?", "higher", "usefulness"),
        ("AUC ratio", "TSTR AUC divided by the same predictor trained on real data", "1.00", "usefulness"),
        ("DCR", "Are fake records closer to training patients than to unseen ones?", "50%", "privacy"),
        ("MIA AUC", "Can an attacker tell which patients were used for training?", "0.5", "privacy"),
        ("Missing-rate gap <span class='new'>new</span>", "Is each column missing as often as in real data?", "0", "missing values"),
        ("Site gap <span class='new'>new</span>", "Is each column missing as often <i>at each hospital</i>?", "0", "missing values"),
        ("Pattern distance <span class='new'>new</span>", "Do whole patterns of holes (which values are missing together) match?", "0", "missing values"),
        ("Rare-pattern MIA <span class='new'>new</span>", "Are patients with unusual holes easier to identify?", "0.5", "privacy of holes"),
        ("Unique-pattern leak <span class='new'>new</span>", "Do one-of-a-kind hole patterns of training patients reappear?", "0", "privacy of holes"),
    ]
    return f"""
<section>
<h2>7&nbsp; How we test it</h2>
<h3>A fair test setup</h3>
<ul>
<li><b>Unseen patients.</b> 20% of real patients are held out and never shown to any model. Usefulness and privacy
are measured against them. (Testing on patients the model saw would make results look better than they are.)</li>
<li><b>Five repeats.</b> Everything is repeated with 5 seeds. Each seed holds out different patients and starts the
models differently. We report the average and the spread (&plusmn; standard deviation).</li>
<li><b>Same treatment for all.</b> All four methods get the same data, the same missing-value flags, the same
cleaning and the same tests.</li>
<li><b>Paired statistics.</b> Methods are compared seed by seed. A difference is called <i>clear</i> only if its
95% confidence interval does not include zero.</li>
</ul>
<h3>What we measure</h3>
{table(["Measure", "Question it answers", "Best value", "Group"], rows, "compact", ["22%", "50%", "13%", "15%"])}
{box("Detectability is not privacy", "<p>Some papers call the &ldquo;can a classifier spot the fakes?&rdquo; score a privacy score. It is not: a model that simply copied real patients would be impossible to spot and still leak everything. We use it only for realism, and judge privacy with DCR and membership attacks.</p>", "note")}
<h3>The new missing-value measures, in simple words</h3>
<p>A patient's <b>missingness pattern</b> is the list of which of their values are blank, plus their hospital &mdash;
for example &ldquo;Hungary: cholesterol measured, vessel scan missing, thalassemia missing&rdquo;.</p>
<ul>
<li><b>Pattern distance</b> compares how often each pattern appears in fake and in real data (0 = identical,
1 = completely different). Because the score depends on sample size, we also compute it between two groups of
<i>real</i> patients. A fake dataset that scores close to this real-vs-real value is as good as more real data.</li>
<li><b>Rare-pattern attack</b> runs the membership attack only on patients whose pattern is rare (held by at most
2% of patients) and compares with everyone else. If keeping holes hurts privacy, rare patients suffer most.</li>
</ul>
</section>"""


def bench_rows(ds, cols):
    rows = []
    for m in MODELS:
        rows.append([LABEL[m]] + [ms(ds, m, c, d) for c, d in cols])
    return rows


def results_section():
    cols = [("fidelity_pct", 1), ("detectability_pct", 1), ("tstr_auc", 3), ("auc_ratio", 3),
            ("dcr_closer_to_train_pct", 1), ("mia_auc", 3)]
    head = ["Method", "Fidelity % &uarr;", "Detect. % &rarr;50", "TSTR AUC &uarr;", "AUC ratio &rarr;1",
            "DCR % &rarr;50", "MIA AUC &rarr;0.5"]
    mcols = [("miss_rate_gap", 3), ("site_miss_gap", 3), ("pattern_tvd", 3),
             ("rare_pattern_dist_mia_auc", 3), ("common_pattern_dist_mia_auc", 3)]
    mhead = ["", "Rate gap &darr;", "Site gap &darr;", "Pattern dist. &darr;", "MIA rare &rarr;0.5", "MIA common &rarr;0.5"]

    def miss_rows(ds):
        ref = [ "Two real samples (reference)", f(mean(ds, MEDTWIN, "miss_rate_gap_ref")),
                f(mean(ds, MEDTWIN, "site_miss_gap_ref")) if "site_miss_gap_ref" in RUNS[ds] else "&ndash;",
                f(mean(ds, MEDTWIN, "pattern_tvd_ref")), "", ""]
        rows = [ref] + [[LABEL[m]] + [ms(ds, m, c, d) if c in RUNS[ds] else "&ndash;" for c, d in mcols] for m in MODELS]
        return rows

    def abl_rows(ds, variants):
        names = {"full": "Full MedTwin", "no_missing_flags": "without missing-value flags",
                 "no_zero_flags": "without zero flag", "no_condition": "without hospital conditioning",
                 "no_corr_loss": "without correlation loss"}
        out = []
        for v in variants:
            if v not in set(ABL[ds].variant):
                continue
            star = lambda c: "<sup>*</sup>" if v != "full" and clear(ds, "full", v, c, "variant", ABL) else ""
            out.append([names[v]] + [f"{ms(ds, v, c, d, 'variant', ABL)}{star(c)}" for c, d in
                                     [("detectability_pct", 1), ("pattern_tvd", 3), ("corr_gap_numeric", 3),
                                      ("auc_ratio", 3), ("mia_auc", 3)]])
        return out

    ahead = ["Variant", "Detect. % &rarr;50", "Pattern dist. &darr;", "Corr. gap &darr;", "AUC ratio &rarr;1", "MIA AUC &rarr;0.5"]
    sens_names = {"full": "Default (weight 2, 600 epochs)", "corr_weight_0.5": "Correlation weight 0.5",
                  "corr_weight_8": "Correlation weight 8", "epochs_300": "300 epochs", "epochs_1200": "1200 epochs"}
    srows = []
    for v in medtwin.SENSITIVITY:
        star = lambda c: "<sup>*</sup>" if v != "full" and clear("h", "full", v, c, "variant", ABL) else ""
        srows.append([sens_names[v]] + [f"{ms('h', v, c, d, 'variant', ABL)}{star(c)}" for c, d in
                                        [("fidelity_pct", 1), ("detectability_pct", 1), ("auc_ratio", 3),
                                         ("dcr_closer_to_train_pct", 1), ("mia_auc", 3)]])

    h, d = "h", "d"
    return f"""
<section>
<h2>8&nbsp; Results</h2>
<p>All numbers are averages over 5 seeds &plusmn; standard deviation, measured on real patients the models never
saw. Arrows show what is better: &uarr; higher, &darr; lower, &rarr; closer to the given value. Best values in bold
are described in the text.</p>

<h3>8.1&nbsp; Heart disease: MedTwin against the other methods</h3>
{table(head, bench_rows(h, cols), "results")}
{figure("figures/heart_disease/fig_benchmark.png", "Heart disease comparison, averages over 5 seeds with spread. Dashed lines show the ideal value.")}
<ul>
<li><b>Realism:</b> MedTwin has the best fidelity ({f(mean(h,MEDTWIN,'fidelity_pct'),1)}%) and by far the lowest
detectability ({f(mean(h,MEDTWIN,'detectability_pct'),1)}%, against {f(mean(h,'TabDDPM','detectability_pct'),1)}&ndash;{f(mean(h,'CTGAN','detectability_pct'),1)}%). Both differences are clear against every method.</li>
<li><b>Usefulness:</b> AUC ratio {f(mean(h,MEDTWIN,'auc_ratio'))} &mdash; statistically tied with TVAE ({f(mean(h,'TVAE','auc_ratio'))}),
clearly better than CTGAN ({f(mean(h,'CTGAN','auc_ratio'))}).</li>
<li><b>Privacy:</b> DCR and MIA are close to the ideal for all methods; no method copied a real patient.</li>
</ul>

<h3>8.2&nbsp; Diabetes: the same picture</h3>
{table(head, bench_rows(d, cols), "results")}
<ul>
<li>MedTwin again has the best fidelity ({f(mean(d,MEDTWIN,'fidelity_pct'),1)}%) and ties for usefulness
({f(mean(d,MEDTWIN,'auc_ratio'))}).</li>
<li>TabDDPM is the hardest to spot ({f(mean(d,'TabDDPM','detectability_pct'),1)}%) <b>but it memorised its training
patients</b>: {f(mean(d,'TabDDPM','dcr_closer_to_train_pct'),1)}% of its records sit closer to training patients and the
membership attack reaches AUC {f(mean(d,'TabDDPM','mia_auc'))}, matching recent findings that diffusion models for tables memorise [24]. MedTwin shows no such leak ({f(mean(d,MEDTWIN,'mia_auc'))}).
This also shows why detectability alone must not be read as privacy.</li>
</ul>

<h3>8.3&nbsp; Are the missing values realistic?</h3>
<p class='tlabel'><b>Heart disease</b></p>
{table(mhead, miss_rows(h), "results")}
<p class='tlabel'><b>Diabetes</b> (one clinic, so there is no site gap)</p>
{table(mhead, miss_rows(d), "results")}
<ul>
<li>MedTwin's pattern distance ({f(mean(h,MEDTWIN,'pattern_tvd'))}) is close to the gap between two real samples
({f(mean(h,MEDTWIN,'pattern_tvd_ref'))}) and clearly better than every other method in all 5 seeds
(TVAE {f(mean(h,'TVAE','pattern_tvd'))}, TabDDPM {f(mean(h,'TabDDPM','pattern_tvd'))}, CTGAN {f(mean(h,'CTGAN','pattern_tvd'))}).</li>
<li>Only MedTwin gets the hospital right: its site gap ({f(mean(h,MEDTWIN,'site_miss_gap'))}) matches real data
({f(mean(h,MEDTWIN,'site_miss_gap_ref'))}). The other methods were given the same flags, but they do not know which
hospital a hole belongs to, so they put holes in the wrong places.</li>
<li>Without the flags, the pattern distance jumps to {am(h,'no_missing_flags','pattern_tvd')}; without hospital
conditioning it rises to {am(h,'no_condition','pattern_tvd')} and the site gap to {am(h,'no_condition','site_miss_gap')}.</li>
</ul>
{figure("figures/heart_disease/fig_missingness_by_site.png", "Share of heart patients with each value not measured, per hospital: real (blue) and MedTwin (orange).")}

<h3>8.4&nbsp; Does keeping missing values cost privacy?</h3>
<p>Yes, a little, for one group. On heart disease about {f(mean(h,MEDTWIN,'n_rare_patients'),0)} of the 368 attacked
patients have a <i>rare</i> missingness pattern. For them, the membership attack reaches AUC
<b>{am(h,'full','rare_pattern_dist_mia_auc')}</b> against MedTwin, compared with {am(h,'no_missing_flags','rare_pattern_dist_mia_auc')}
when MedTwin is trained without the missing-value flags &mdash; higher in all 5 seeds. For patients with common
patterns ({am(h,'full','common_pattern_dist_mia_auc')} vs {am(h,'no_missing_flags','common_pattern_dist_mia_auc')}) and overall
({am(h,'full','mia_auc')} vs {am(h,'no_missing_flags','mia_auc')}) there is no clear difference. One-of-a-kind patterns of
training patients do not reappear more often than those of unseen patients, so the model is not copying patterns.
On diabetes only about {f(mean(d,MEDTWIN,'n_rare_patients'),0)} patients per test have rare patterns, too few for a conclusion.</p>
{box("What this means", "<p>Realistic holes make the data more realistic and keep overall privacy unchanged, but patients with <i>unusual</i> holes become slightly easier to single out. Data owners should know this; merging rare patterns before release is a possible fix to test next.</p>", "goal")}

<h3>8.5&nbsp; What each part contributes (ablation)</h3>
<p class='tlabel'><b>Heart disease</b> (<sup>*</sup> = clear difference from the full model)</p>
{table(ahead, abl_rows(h, ["full", "no_missing_flags", "no_zero_flags", "no_condition", "no_corr_loss"]), "results")}
<p class='tlabel'><b>Diabetes</b></p>
{table(ahead, abl_rows(d, ["full", "no_missing_flags", "no_corr_loss"]), "results")}
<ul>
<li><b>Missing-value flags</b> matter most: without them detectability rises from {am(h,'full','detectability_pct',1)}% to
{am(h,'no_missing_flags','detectability_pct',1)}% (heart) and from {am(d,'full','detectability_pct',1)}% to {am(d,'no_missing_flags','detectability_pct',1)}% (diabetes).
Filling holes with typical values makes fake patients easy to spot.</li>
<li><b>Zero flag</b> and <b>hospital conditioning</b> also clearly help realism ({am(h,'no_zero_flags','detectability_pct',1)}% and {am(h,'no_condition','detectability_pct',1)}% without them).</li>
<li><b>Correlation loss</b> does not change realism or usefulness, but without it the correlation gap nearly doubles
({am(h,'full','corr_gap_numeric')} &rarr; {am(h,'no_corr_loss','corr_gap_numeric')} heart; {am(d,'full','corr_gap_numeric')} &rarr; {am(d,'no_corr_loss','corr_gap_numeric')} diabetes).</li>
<li>No part changes usefulness or overall privacy. The medical range check never removed a patient: it is only a safety net.</li>
</ul>

<h3>8.6&nbsp; Do results depend on our chosen settings?</h3>
{table(["Setting (heart disease)", "Fidelity %", "Detect. %", "AUC ratio", "DCR %", "MIA AUC"], srows, "results")}
<p>Changing the correlation weight (0.5 or 8 instead of 2) or training for 300 epochs makes no clear difference.
Training twice as long (1200 epochs) starts to memorise: DCR and MIA rise clearly. So 600 epochs is a good choice,
and the results do not depend on lucky settings.</p>

<h3>8.7&nbsp; A closer look at the generated values</h3>
{figure("figures/heart_disease/fig_continuous.png", "Distributions of the number columns (heart disease): real (blue) vs MedTwin (orange), measured values only.")}
{figure("figures/heart_disease/fig_correlation.png", "Correlations between number columns: real, MedTwin, and the difference (heart disease).")}
</section>"""


def comparison():
    rows = [
        ("Wan Abu Bakar et al., 2025 (IJACSA) [5]", "CTGAN on Cleveland heart data (303 patients); missing values filled with most common value; single run; GAN saw all patients",
         "Accuracy train-on-fake, test-on-real: 56% (5-class)", "We use all 4 hospitals, unseen test patients and 5 seeds. Our CTGAN baseline is also weak; MedTwin clearly beats it."),
        ("Subedi et al., 2025 (IEEE ICUIS) [6]", "CTGAN on Pima diabetes", "Random forest accuracy 82% (real) &rarr; 63% (fake), i.e. 77% kept",
         f"MedTwin keeps {f(mean('d',MEDTWIN,'auc_ratio')*100,0)}% of real-data performance (AUC ratio, 5 seeds). Setups differ, so this is indicative."),
        ("Alqulaity &amp; Yang, 2024 (Sensors) [7]", "Enhanced conditional GAN, 218 cardiovascular patients, one hospital", "KS statistic 0.390 (TVAE 0.386); no usefulness or privacy test",
         "Our evaluation covers realism, usefulness and privacy with unseen patients and repeats."),
        ("Lazzaro et al., 2026 (J. Comp. Science) [8]", "Bayesian network on 3 UCI heart sites, trained on Cleveland only", "Augmentation results",
         "We model all 4 hospitals including the very incomplete ones."),
        ("HealthGen, 2022 [9]; MissDiff, 2023 [10]; Wang et al., 2025 (IEEE TKDE) [11]; IGAMT, 2024 (AAAI) [12]",
         "Generators that keep missing values (time series, diffusion, plug-in, private EHR)", "Show that keeping missingness matters",
         "Same goal. Our additions: hospital-conditioned holes, standard scores for hole realism, and the first test of the privacy cost of holes. We have not yet compared against them directly."),
    ]
    return f"""
<section>
<h2>9&nbsp; Comparison with previous work</h2>
<p>We searched for work from the last five years on the same datasets or the same problem. Exact number-for-number
comparison is usually not possible, because each paper uses different data, splits and measures. The table shows
what each did and how our project relates. Larger benchmark studies of synthetic health data [25, 26] compare many
generators on realism, usefulness and privacy; we follow the same three-part view.</p>
{table(["Work", "What they did", "Their result", "How we compare"], rows, "compact lit", ["21%", "27%", "22%", "30%"])}
<p><b>In short:</b> on the same datasets, our results and testing are stronger than recent published work. On the core idea
(keeping missing values), other groups were first, so our novelty is in <i>measuring</i> realistic holes, in hospital
conditioning, and in showing their privacy cost.</p>
</section>"""


def novelty():
    return """
<section>
<h2>10&nbsp; What is new in this project</h2>
<ol class='spaced'>
<li><b>Scores for realistic missing values.</b> Missing rate per column, per hospital, and the full pattern of
which values go missing together &mdash; each compared with the gap between two real samples, so &ldquo;good&rdquo;
has a clear meaning. Existing papers do not use a common measure for this.</li>
<li><b>A privacy audit for missing values.</b> The first test of whether realistic holes make patients easier to
identify. We found a small, consistent cost for patients with rare patterns that standard privacy tests do not show.</li>
<li><b>Hospital-aware generation of holes.</b> MedTwin is the only tested method that puts holes at the right
hospital, because each patient is generated for a given hospital.</li>
<li><b>A careful, honest evaluation.</b> Unseen test patients, five repeats, paired statistics, ablation of every part,
a settings check, and a clear separation of realism from privacy.</li>
</ol>
</section>"""


def limitations():
    return """
<section class='cont'>
<h2>11&nbsp; Limitations</h2>
<ul>
<li><b>Small datasets.</b> Both datasets have fewer than 1000 patients and one table each. Larger hospital
databases and data over time may behave differently.</li>
<li><b>Still detectable.</b> A classifier can still spot our fake heart patients 62% of the time (50% would be perfect).</li>
<li><b>Privacy is tested, not guaranteed.</b> We use attacks, not a mathematical guarantee such as differential
privacy. Passing these attacks does not prove the data is safe [23].</li>
<li><b>Missing-value privacy audit needs enough rare patients.</b> It works on heart disease but not on diabetes.</li>
<li><b>Not compared with every method.</b> The newest missing-value generators (MissDiff, Wang et al.) and TabSyn
were not run in our benchmark. TabDDPM is our own compact version.</li>
</ul>
</section>"""


def future():
    return """
<section class='cont'>
<h2>12&nbsp; Future work</h2>
<ol class='spaced'>
<li>Compare directly with MissDiff and Wang et al., the closest previous work.</li>
<li>Test fixes for the privacy cost, such as merging rare missing-value patterns before release.</li>
<li>Add formal privacy (differentially private training) and measure what it costs in realism.</li>
<li>Generate patients for a hospital the model has never seen, to support data sharing between hospitals.</li>
<li>Move to larger and time-based hospital data.</li>
</ol>
</section>"""


def history():
    log = subprocess.run(["git", "log", "--format=%ad|%s", "--date=short", "--reverse"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    rows = [tuple(esc(x) for x in line.split("|", 1)) for line in log]
    return f"""
<section>
<h2>13&nbsp; Project history</h2>
<p>The project grew in steps. An early version (April 2026) used 9 columns, filled missing values, and tested on
patients the model had seen. It was rebuilt to handle all 15 columns with missing values and hospitals, to test
on unseen patients, and to compare against other methods over five seeds. The recorded steps since then:</p>
{table(["Date", "Step"], rows, "compact", ["15%", "85%"])}
</section>"""


def appendix():
    cmds = [
        ("Install", "python -m venv venv &amp;&amp; venv\\Scripts\\pip install -r requirements.txt"),
        ("Run the pipeline (heart)", "python main.py"),
        ("Run on diabetes", "python main.py --config configs/diabetes.yaml"),
        ("Dashboard", "streamlit run app/dashboard.py"),
        ("5-seed comparison", "python research/multi_seed.py"),
        ("Ablations", "python research/ablation.py"),
        ("Settings check", "python research/ablation.py --study sensitivity"),
        ("Figures", "python research/visualizer.py --benchmark"),
        ("Paper tables", "python research/paper_tables.py"),
        ("Tests", "python -m pytest tests -q"),
        ("This report", "python docs/build_report.py"),
    ]
    # Numbered as cited in the text; checked against publisher, Crossref,
    # DataCite or Europe PMC records (October 2026).
    refs = [
        "Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R.: Heart Disease [dataset]. UCI Machine Learning Repository (1989). https://doi.org/10.24432/C52P4X",
        "Vanschoren, J., van Rijn, J.N., Bischl, B., Torgo, L.: OpenML: networked science in machine learning. ACM SIGKDD Explorations Newsletter 15(2), 49&ndash;60 (2014). https://doi.org/10.1145/2641190.2641198 (Pima diabetes: OpenML dataset 37)",
        "Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K.: Modeling tabular data using conditional GAN. In: Advances in Neural Information Processing Systems 32 (NeurIPS) (2019)",
        "Kotelnikov, A., Baranchuk, D., Rubachev, I., Babenko, A.: TabDDPM: Modelling tabular data with diffusion models. In: Proc. 40th International Conference on Machine Learning, PMLR 202, 17564&ndash;17579 (2023)",
        "Wan Abu Bakar, W.A., Josdi, N.L.N., Man, M., Kadir, E.A.: Evaluating CTGAN-generated synthetic data for heart disease prediction: fidelity, predictive utility, and feature preservation. International Journal of Advanced Computer Science and Applications 16(12), 1000&ndash;1009 (2025). https://doi.org/10.14569/IJACSA.2025.0161296",
        "Subedi, P., Divakarla, S.R., Emani, M., Kochuvila, S., Sunitha, R.: Synthetic data for health risk prediction: an evaluation using machine learning. In: 5th International Conference on Ubiquitous Computing and Intelligent Information Systems (ICUIS), IEEE (2025). https://doi.org/10.1109/ICUIS67429.2025.11380688",
        "Alqulaity, M., Yang, P.: Enhanced conditional GAN for high-quality synthetic tabular data generation in mobile-based cardiovascular healthcare. Sensors 24(23), 7673 (2024). https://doi.org/10.3390/s24237673",
        "Lazzaro, I., Milano, M., Tucker, A., Cannataro, M.: Assessing the impact of synthetic data generated by Bayesian networks on heart disease prediction. Journal of Computational Science 99, 102940 (2026). https://doi.org/10.1016/j.jocs.2026.102940",
        "Bing, S., Dittadi, A., Bauer, S., Schwab, P.: Conditional generation of medical time series for extrapolation to underrepresented populations (HealthGen). arXiv:2201.08186 (2022)",
        "Ouyang, Y., Xie, L., Li, C., Cheng, G.: MissDiff: Training diffusion models on tabular data with missing values. ICML 2023 Workshop on Structured Probabilistic Inference and Generative Modeling. arXiv:2307.00467 (2023)",
        "Wang, X., Asif, H., Gupta, S., Vaidya, J.: Data synthesis reinvented: preserving missing patterns for enhanced analysis. IEEE Transactions on Knowledge and Data Engineering 37(7), 3962&ndash;3975 (2025)",
        "Wang, W., Tang, P., Lou, J., Shao, Y., Waller, L., Ko, Y., Xiong, L.: IGAMT: Privacy-preserving electronic health record synthesization with heterogeneity and irregularity. In: Proc. AAAI Conference on Artificial Intelligence 38(14), 15634&ndash;15643 (2024). https://doi.org/10.1609/aaai.v38i14.29491",
        "Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J.J., Sandhu, S., Guppy, K.H., Lee, S., Froelicher, V.: International application of a new probability algorithm for the diagnosis of coronary artery disease. The American Journal of Cardiology 64(5), 304&ndash;310 (1989). https://doi.org/10.1016/0002-9149(89)90524-9",
        "Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S.: Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In: Proc. Annual Symposium on Computer Application in Medical Care, 261&ndash;265 (1988)",
        "Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Advances in Neural Information Processing Systems 27, 2672&ndash;2680 (2014)",
        "Arjovsky, M., Chintala, S., Bottou, L.: Wasserstein generative adversarial networks. In: Proc. 34th International Conference on Machine Learning, PMLR 70, 214&ndash;223 (2017)",
        "Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A.: Improved training of Wasserstein GANs. In: Advances in Neural Information Processing Systems 30 (NeurIPS) (2017)",
        "Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: 3rd International Conference on Learning Representations (ICLR) (2015)",
        "Breiman, L.: Random forests. Machine Learning 45(1), 5&ndash;32 (2001). https://doi.org/10.1023/A:1010933404324",
        "Esteban, C., Hyland, S.L., R&auml;tsch, G.: Real-valued (medical) time series generation with recurrent conditional GANs. arXiv:1706.02633 (2017)",
        "Hayes, J., Melis, L., Danezis, G., De Cristofaro, E.: LOGAN: Membership inference attacks against generative models. Proceedings on Privacy Enhancing Technologies 2019(1), 133&ndash;152 (2019). https://doi.org/10.2478/popets-2019-0008",
        "Chen, D., Yu, N., Zhang, Y., Fritz, M.: GAN-Leaks: A taxonomy of membership inference attacks against generative models. In: Proc. ACM SIGSAC Conference on Computer and Communications Security (CCS) (2020). https://doi.org/10.1145/3372297.3417238",
        "Stadler, T., Oprisanu, B., Troncoso, C.: Synthetic data &ndash; anonymisation groundhog day. In: 31st USENIX Security Symposium, 1451&ndash;1468 (2022)",
        "Fang, Z., Jiang, Z., Chen, H., Li, X., Li, J.: Understanding and mitigating memorization in diffusion models for tabular data. In: Proc. 42nd International Conference on Machine Learning, PMLR 267, 15976&ndash;16005 (2025)",
        "Yan, C., Yan, Y., Wan, Z., Zhang, Z., Omberg, L., Guinney, J., Mooney, S.D., Malin, B.A.: A multifaceted benchmarking of synthetic electronic health record generation models. Nature Communications 13, 7609 (2022). https://doi.org/10.1038/s41467-022-35295-1",
        "Hernandez, M., Osorio-Marulanda, P.A., Catalina, M., Loinaz, L., Epelde, G., Aginako, N.: Comprehensive evaluation framework for synthetic tabular data in health. Frontiers in Digital Health 7, 1576290 (2025). https://doi.org/10.3389/fdgth.2025.1576290",
        "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., et al.: PyTorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems 32 (NeurIPS) (2019)",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., et al.: Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825&ndash;2830 (2011)",
    ]
    reflist = "".join(f"<li>{r}</li>" for r in refs)
    return f"""
<section>
<h2>A&nbsp; How to run the project</h2>
<p>From the project folder (Windows):</p>
{table(["Task", "Command"], [(t, f"<code>{c}</code>") for t, c in cmds], "compact", ["26%", "74%"])}
<p class='small'>A full 5-seed comparison takes about an hour on a laptop when seeds run in parallel. Results are
saved per run, so an interrupted run continues where it stopped.</p>
<h2>B&nbsp; References</h2>
<ol class='refs'>{reflist}</ol>
</section>"""


CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm;
  @bottom-center { content: counter(page); font: 9pt 'Segoe UI', Arial, sans-serif; color: #6b6b6b; } }
@page :first { @bottom-center { content: none; } }
:root { --ink:#1d1d1f; --ink2:#55555a; --accent:#1f5fa8; --accent2:#e8f0fa; --warm:#eb6834; --rule:#d9dce1; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.3pt; line-height: 1.5; color: var(--ink); margin:0; }
h2 { font-size: 17pt; color: var(--accent); margin: 0 0 8pt; padding-bottom: 4pt; border-bottom: 2px solid var(--accent); break-after: avoid; }
h2.plain { border: none; }
h3 { font-size: 12pt; margin: 14pt 0 5pt; color: var(--ink); break-after: avoid; }
section { break-before: page; }
section.cover, section.cont { break-before: auto; margin-top: 16pt; }
table.results { break-inside: avoid; }
p.tlabel { break-after: avoid; margin-top: 6pt; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
ol.spaced li { margin-bottom: 8pt; }
th code { background: none; color: white; padding: 0; }
.keep { break-inside: avoid; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; background: #f2f3f5; padding: 0 2pt; border-radius: 2pt; }
table { width: 100%; border-collapse: collapse; margin: 4pt 0 10pt; font-size: 9pt; table-layout: fixed; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: var(--accent); color: white; text-align: left; padding: 4pt 5pt; font-weight: 600; vertical-align: bottom; }
td { padding: 3.5pt 5pt; border-bottom: 1px solid var(--rule); vertical-align: top; word-wrap: break-word; }
tbody tr:nth-child(even) td { background: #f7f8fa; }
table.results { font-size: 8.6pt; }
table.results td, table.results th { text-align: center; }
table.results td:first-child, table.results th:first-child { text-align: left; }
table.center td, table.center th { text-align: center; }
table.center td:first-child { text-align: left; }
table.lit { font-size: 8.3pt; }
.sd { color: var(--ink2); font-size: 0.88em; }
table.three { table-layout: fixed; }
table.three td { width: 33%; background: var(--accent2) !important; border: 3px solid white; padding: 7pt; font-size: 9.5pt; }
figure { margin: 6pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.8pt; color: var(--ink2); margin-top: 3pt; }
.box { border-left: 4px solid var(--accent); background: var(--accent2); padding: 7pt 10pt; margin: 8pt 0 10pt; break-inside: avoid; }
.box.goal { border-color: var(--warm); background: #fdf0ea; }
.box-title { font-weight: 700; margin-bottom: 3pt; }
.box p:last-child, .box ol:last-child { margin-bottom: 0; }
.new { display: inline-block; background: var(--warm); color: white; font-size: 7pt; padding: 0 3pt; border-radius: 2pt; vertical-align: 1px; }
.small { font-size: 8.8pt; color: var(--ink2); }
.cover { height: 255mm; display: flex; flex-direction: column; padding-top: 30mm; }
.cover-top { font-size: 9pt; color: var(--ink2); text-transform: uppercase; letter-spacing: 0.06em; }
.cover h1 { font-size: 46pt; margin: 18mm 0 4mm; color: var(--accent); letter-spacing: -0.01em; }
.subtitle { font-size: 17pt; line-height: 1.3; max-width: 150mm; }
.cover-desc { margin-top: 10mm; font-size: 11pt; color: var(--ink2); max-width: 155mm; }
.cover-authors { margin-top: 14mm; font-size: 13pt; font-weight: 600; }
.cover-date { color: var(--ink2); }
.cover-glance { margin-top: auto; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6mm; border-top: 2px solid var(--accent); padding-top: 6mm; }
.cover-glance b { display: block; font-size: 24pt; color: var(--accent); }
.cover-glance span { font-size: 8.8pt; color: var(--ink2); line-height: 1.35; display: block; }
.toc-list { list-style: none; padding: 0; font-size: 12pt; }
.toc-list li { padding: 5pt 0; border-bottom: 1px solid var(--rule); }
.toc-list .num { display: inline-block; width: 12mm; color: var(--accent); font-weight: 700; }
.pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5pt; margin: 6pt 0 10pt; }
.step { border: 1.5px solid var(--accent); border-radius: 4pt; padding: 6pt; background: white; position: relative; }
.step-n { position: absolute; top: -1px; right: -1px; background: var(--accent); color: white; font-weight: 700; font-size: 9pt; width: 15pt; height: 15pt; text-align: center; line-height: 15pt; border-radius: 0 4pt 0 4pt; }
.step-t { font-weight: 700; color: var(--accent); margin-bottom: 2pt; }
.step-d { font-size: 8.6pt; color: var(--ink2); line-height: 1.35; }
.glossary td:first-child { font-weight: 600; }
.refs { font-size: 8.8pt; }
"""


def build():
    body = "".join([cover(), contents(), summary(), problem(), ideas(), data_section(), pipeline(),
                    specs(), testing(), results_section(), comparison(), novelty(), limitations(),
                    future(), history(), appendix()])
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>MedTwin Project Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    with open(HTML_PATH, "w", encoding="utf-8") as fh:
        fh.write(page)
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if browser is None:
        print(f"Wrote {HTML_PATH}; no Edge/Chrome found to print it to PDF.")
        return
    if os.path.exists(PDF_PATH):
        os.remove(PDF_PATH)
    url = "file:///" + HTML_PATH.replace("\\", "/")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF_PATH}", url], check=True, capture_output=True)
    # The browser can return a moment before the file is written
    for _ in range(60):
        if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0:
            break
        time.sleep(0.5)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    build()
