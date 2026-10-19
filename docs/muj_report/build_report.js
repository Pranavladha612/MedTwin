// Builds the MUJ-format project report (Word) for MedTwin.
//
//   node docs/muj_report/build_report.js
//
// Layout follows the department's template ("Project Report Final"):
// cover, certificate, acknowledgments, abstract, lists of figures and
// tables, contents, abbreviations, chapters 1-6, references (IEEE) and
// annexures. Every result number comes from data.json (prep_data.py).
// Page numbers in the contents and lists are PAGEREF fields; update_fields.ps1
// fills them in with Word and exports the PDF.

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, Bookmark, BorderStyle, Document, Footer, HeadingLevel, ImageRun,
  LevelFormat, NumberFormat, PageBreak, PageNumber, PageReference, Packer, Paragraph,
  ShadingType, Table, TableCell, TableRow, TabStopType, TextRun, VerticalAlign, WidthType,
} = require("docx");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..", "..");
const D = require("./data.json");
const OUT = process.env.MUJ_OUT || path.join(ROOT, "docs", "MedTwin_Project_Report_MUJ.docx");

const MT = "MedTwin WGAN-GP";
const H = D.heart, DB = D.diabetes;
const TEXT_W = 8666; // A4 width 11906 minus left 1800 and right 1440 margins (DXA)
const FONT = "Times New Roman";

// ------------------------------------------------------------- helpers ---
const f = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const m = (st, d = 3) => (st ? f(st.mean, d) : "–");
const ms = (st, d = 3) => (st ? `${f(st.mean, d)} ± ${f(st.sd, d)}` : "–");
const hm = (model, k) => H.models[model][k];
const dm = (model, k) => DB.models[model][k];
const hv = (v, k) => H.variants[v][k];
const dv = (v, k) => DB.variants[v][k];

// IEEE references, numbered in order of first citation
const REFS = {
  janosi: 'A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1989, doi: 10.24432/C52P4X.',
  detrano: 'R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J.-J. Schmid, S. Sandhu, K. H. Guppy, S. Lee, and V. Froelicher, "International application of a new probability algorithm for the diagnosis of coronary artery disease," The American Journal of Cardiology, vol. 64, no. 5, pp. 304–310, 1989, doi: 10.1016/0002-9149(89)90524-9.',
  smith: 'J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," in Proc. Annual Symposium on Computer Application in Medical Care, 1988, pp. 261–265.',
  openml: 'J. Vanschoren, J. N. van Rijn, B. Bischl, and L. Torgo, "OpenML: Networked science in machine learning," ACM SIGKDD Explorations Newsletter, vol. 15, no. 2, pp. 49–60, 2014, doi: 10.1145/2641190.2641198.',
  goncalves: 'A. Goncalves, P. Ray, B. Soper, J. Stevens, L. Coyle, and A. P. Sales, "Generation and evaluation of synthetic patient data," BMC Medical Research Methodology, vol. 20, art. no. 108, 2020, doi: 10.1186/s12874-020-00977-1.',
  yan: 'C. Yan, Y. Yan, Z. Wan, Z. Zhang, L. Omberg, J. Guinney, S. D. Mooney, and B. A. Malin, "A multifaceted benchmarking of synthetic electronic health record generation models," Nature Communications, vol. 13, art. no. 7609, 2022, doi: 10.1038/s41467-022-35295-1.',
  goodfellow: 'I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Advances in Neural Information Processing Systems 27, 2014, pp. 2672–2680.',
  arjovsky: 'M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," in Proc. 34th International Conference on Machine Learning, PMLR, vol. 70, 2017, pp. 214–223.',
  gulrajani: 'I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved training of Wasserstein GANs," in Advances in Neural Information Processing Systems 30, 2017.',
  jang: 'E. Jang, S. Gu, and B. Poole, "Categorical reparameterization with Gumbel-softmax," in Proc. International Conference on Learning Representations (ICLR), 2017.',
  xu: 'L. Xu, M. Skoularidou, A. Cuesta-Infante, and K. Veeramachaneni, "Modeling tabular data using conditional GAN," in Advances in Neural Information Processing Systems 32, 2019.',
  kotelnikov: 'A. Kotelnikov, D. Baranchuk, I. Rubachev, and A. Babenko, "TabDDPM: Modelling tabular data with diffusion models," in Proc. 40th International Conference on Machine Learning, PMLR, vol. 202, 2023, pp. 17564–17579.',
  esteban: 'C. Esteban, S. L. Hyland, and G. Rätsch, "Real-valued (medical) time series generation with recurrent conditional GANs," arXiv:1706.02633, 2017.',
  hayes: 'J. Hayes, L. Melis, G. Danezis, and E. De Cristofaro, "LOGAN: Membership inference attacks against generative models," Proceedings on Privacy Enhancing Technologies, vol. 2019, no. 1, pp. 133–152, 2019, doi: 10.2478/popets-2019-0008.',
  chen: 'D. Chen, N. Yu, Y. Zhang, and M. Fritz, "GAN-Leaks: A taxonomy of membership inference attacks against generative models," in Proc. ACM SIGSAC Conference on Computer and Communications Security (CCS), 2020, doi: 10.1145/3372297.3417238.',
  stadler: 'T. Stadler, B. Oprisanu, and C. Troncoso, "Synthetic data – anonymisation groundhog day," in Proc. 31st USENIX Security Symposium, 2022, pp. 1451–1468.',
  alaa: 'A. Alaa, B. van Breugel, E. Saveliev, and M. van der Schaar, "How faithful is your synthetic data? Sample-level metrics for evaluating and auditing generative models," in Proc. 39th International Conference on Machine Learning, PMLR, vol. 162, 2022, pp. 290–306.',
  kingma: 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. 3rd International Conference on Learning Representations (ICLR), 2015.',
  breiman: 'L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.',
  paszke: 'A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems 32, 2019.',
  pedregosa: 'F. Pedregosa, G. Varoquaux, A. Gramfort et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.',
  choi: 'E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, and J. Sun, "Generating multi-label discrete patient records using generative adversarial networks," in Proc. 2nd Machine Learning for Healthcare Conference, PMLR, vol. 68, 2017, pp. 286–305.',
  zhao: 'Z. Zhao, A. Kunar, R. Birke, and L. Y. Chen, "CTAB-GAN: Effective table data synthesizing," in Proc. 13th Asian Conference on Machine Learning, PMLR, vol. 157, 2021, pp. 97–112.',
  zhang: 'H. Zhang, J. Zhang, B. Srinivasan, Z. Shen, X. Qin, C. Faloutsos, H. Rangwala, and G. Karypis, "Mixed-type tabular data synthesis with score-based diffusion in latent space," in Proc. International Conference on Learning Representations (ICLR), 2024.',
  ehrsafe: 'J. Yoon, M. Mizrahi, N. F. Ghalaty, T. Jarvinen, A. S. Ravi, P. Brune, F. Kong, D. Anderson, G. Lee, A. Meir, F. Bandukwala, E. Kanal, S. Ö. Arık, and T. Pfister, "EHR-Safe: Generating high-fidelity and privacy-preserving synthetic electronic health records," npj Digital Medicine, vol. 6, art. no. 141, 2023, doi: 10.1038/s41746-023-00888-7.',
  gain: 'J. Yoon, J. Jordon, and M. van der Schaar, "GAIN: Missing data imputation using generative adversarial nets," in Proc. 35th International Conference on Machine Learning, PMLR, vol. 80, 2018, pp. 5689–5698.',
  misgan: 'S. C.-X. Li, B. Jiang, and B. M. Marlin, "MisGAN: Learning from incomplete data with generative adversarial networks," in Proc. International Conference on Learning Representations (ICLR), 2019.',
  healthgen: 'S. Bing, A. Dittadi, S. Bauer, and P. Schwab, "Conditional generation of medical time series for extrapolation to underrepresented populations," arXiv:2201.08186, 2022.',
  missdiff: 'Y. Ouyang, L. Xie, C. Li, and G. Cheng, "MissDiff: Training diffusion models on tabular data with missing values," ICML 2023 Workshop on Structured Probabilistic Inference and Generative Modeling, arXiv:2307.00467, 2023.',
  reinvented: 'X. Wang, H. Asif, S. Gupta, and J. Vaidya, "Data synthesis reinvented: Preserving missing patterns for enhanced analysis," IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 7, pp. 3962–3975, 2025.',
  igamt: 'W. Wang, P. Tang, J. Lou, Y. Shao, L. Waller, Y. Ko, and L. Xiong, "IGAMT: Privacy-preserving electronic health record synthesization with heterogeneity and irregularity," in Proc. AAAI Conference on Artificial Intelligence, vol. 38, no. 14, 2024, pp. 15634–15643, doi: 10.1609/aaai.v38i14.29491.',
  wanabubakar: 'W. A. Wan Abu Bakar, N. L. N. Josdi, M. Man, and E. A. Kadir, "Evaluating CTGAN-generated synthetic data for heart disease prediction: Fidelity, predictive utility, and feature preservation," International Journal of Advanced Computer Science and Applications, vol. 16, no. 12, pp. 1000–1009, 2025, doi: 10.14569/IJACSA.2025.0161296.',
  subedi: 'P. Subedi, S. R. Divakarla, M. Emani, S. Kochuvila, and R. Sunitha, "Synthetic data for health risk prediction: An evaluation using machine learning," in Proc. 5th International Conference on Ubiquitous Computing and Intelligent Information Systems (ICUIS), IEEE, 2025, doi: 10.1109/ICUIS67429.2025.11380688.',
  alqulaity: 'M. Alqulaity and P. Yang, "Enhanced conditional GAN for high-quality synthetic tabular data generation in mobile-based cardiovascular healthcare," Sensors, vol. 24, no. 23, art. no. 7673, 2024, doi: 10.3390/s24237673.',
  lazzaro: 'I. Lazzaro, M. Milano, A. Tucker, and M. Cannataro, "Assessing the impact of synthetic data generated by Bayesian networks on heart disease prediction," Journal of Computational Science, vol. 99, art. no. 102940, 2026, doi: 10.1016/j.jocs.2026.102940.',
  hernandez: 'M. Hernandez, P. A. Osorio-Marulanda, M. Catalina, L. Loinaz, G. Epelde, and N. Aginako, "Comprehensive evaluation framework for synthetic tabular data in health: Fidelity, utility and privacy analysis of generative models with and without privacy guarantees," Frontiers in Digital Health, vol. 7, art. no. 1576290, 2025, doi: 10.3389/fdgth.2025.1576290.',
  synthcity: 'Z. Qian, R. Davis, and M. van der Schaar, "Synthcity: A benchmark framework for diverse use cases of tabular synthetic data," in Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track, 2023.',
  fang: 'Z. Fang, Z. Jiang, H. Chen, X. Li, and J. Li, "Understanding and mitigating memorization in diffusion models for tabular data," in Proc. 42nd International Conference on Machine Learning, PMLR, vol. 267, 2025, pp. 15976–16005.',
};
const ORDER = [];
const cite = (...keys) =>
  "[" + keys.map((k) => {
    if (!REFS[k]) throw new Error("unknown reference " + k);
    if (!ORDER.includes(k)) ORDER.push(k);
    return ORDER.indexOf(k) + 1;
  }).join(", ") + "]";

// Inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, mt;
  while ((mt = re.exec(text))) {
    if (mt.index > last) out.push(new TextRun({ text: text.slice(last, mt.index), ...base }));
    const t = mt[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = mt.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 0, before: opts.before ?? 0, line: opts.line ?? 240 },
  indent: opts.indent,
  keepNext: opts.keepNext,
  pageBreakBefore: opts.pageBreakBefore,
  children: typeof text === "string" ? runs(text, opts.run || {}) : text,
});
const C = (text, opts = {}) => P(text, { ...opts, align: AlignmentType.CENTER });

let listInstance = 0;
const bullets = (items, level = 0) => items.map((t) => new Paragraph({
  numbering: { reference: "bullets", level },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 0, line: 240 },
  children: runs(t),
}));
function numbered(items) {
  const inst = ++listInstance;
  return items.map((t) => new Paragraph({
    numbering: { reference: "numbers", level: 0, instance: inst },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 0, line: 240 },
    children: runs(t),
  }));
}

// Headings with bookmarks for the contents table
const CONTENTS = []; // {level, label, title, id}
let bmCount = 0;
const newId = (p) => `${p}_${++bmCount}`;
// Chapter pages start with the title alone (template: "INTRODUCTION",
// "Results and Analysis"); the contents table carries "Chapter n".
function chapter(num, title) {
  const id = newId("ch");
  CONTENTS.push({ level: 0, label: `Chapter ${num}`, title: title.toUpperCase(), id });
  return [new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, pageBreakBefore: true,
    spacing: { after: 300 }, children: [new Bookmark({ id, children: [new TextRun({ text: title })] })] })];
}
// Sections are written "1.1) Title", 16 pt bold, centred (template)
function section(num, title) {
  const id = newId("sec");
  CONTENTS.push({ level: 1, label: num, title, id });
  return new Paragraph({ heading: HeadingLevel.HEADING_2, alignment: AlignmentType.CENTER, spacing: { before: 240, after: 160 },
    keepNext: true, children: [new Bookmark({ id, children: [new TextRun({ text: `${num}) ${title}` })] })] });
}
// A section that opens its own page under a plain title (template's last
// chapter: "Conclusions", then "Future Scope of Work")
function pageSection(num, title, chapterEntry) {
  const id = newId("sec");
  const ids = [];
  if (chapterEntry) {
    const cid = newId("ch");
    CONTENTS.push({ level: 0, label: chapterEntry.label, title: chapterEntry.title, id: cid });
    ids.push(cid);
  }
  CONTENTS.push({ level: 1, label: num, title, id });
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, pageBreakBefore: true,
    spacing: { after: 300 }, children: [
      ...ids.map((cid) => new Bookmark({ id: cid, children: [new TextRun({ text: "" })] })),
      new Bookmark({ id, children: [new TextRun({ text: title })] })] });
}
const sub = (title) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 180, after: 80 }, keepNext: true,
  children: [new TextRun({ text: title })] });
function unnumbered(title, opts = {}) {
  const id = newId("fm");
  CONTENTS.push({ level: opts.level ?? 0, label: "", title: opts.toc ?? title, id, front: opts.front });
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, pageBreakBefore: opts.pageBreak !== false,
    spacing: { after: 300 }, children: [new Bookmark({ id, children: [new TextRun({ text: title })] })] });
}

// Tables
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, opts = {}) {
  const children = Array.isArray(text) ? text : [new Paragraph({
    alignment: opts.align || AlignmentType.LEFT,
    spacing: { after: 0, line: Math.round((opts.size || 24) * 11.17), lineRule: "exact" },
    children: typeof text === "string" ? runs(text, { size: opts.size || 24, bold: opts.bold }) : text,
  })];
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    margins: { top: opts.pad ?? 50, bottom: opts.pad ?? 50, left: 90, right: 90 },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    children,
  });
}
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const align = (i) => (opts.center && i > 0 ? AlignmentType.CENTER : AlignmentType.LEFT);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { bold: true, fill: "D9E2F3", align: align(i), size: opts.size || 24 })) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(String(c), widths[i], { align: align(i), size: opts.size, fill: r._fill })) })),
    ],
  });
}
const scale = (ws) => { const t = ws.reduce((a, b) => a + b, 0); return ws.map((w) => Math.round((w / t) * TEXT_W)); };

// Figures and tables with numbered, bookmarked captions
const FIGURES = [], TABLES = [];
function imageSize(file) {
  const b = fs.readFileSync(file);
  if (b[0] === 0x89) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  let i = 2; // JPEG
  while (i < b.length) {
    const marker = b[i + 1], len = b.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7) };
    i += 2 + len;
  }
  throw new Error("unknown image size " + file);
}
function image(file, widthPx) {
  const { w, h } = imageSize(file);
  return new ImageRun({ type: file.endsWith(".png") ? "png" : "jpg", data: fs.readFileSync(file),
    transformation: { width: widthPx, height: Math.round((widthPx * h) / w) } });
}
function figure(chap, rel, caption, widthPx = 560) {
  const n = `${chap}.${FIGURES.filter((x) => x.chap === chap).length + 1}`;
  const id = newId("fig");
  FIGURES.push({ chap, n, caption, id });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [image(path.join(ROOT, rel), widthPx)] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [
      new Bookmark({ id, children: [new TextRun({ text: `Figure ${n}: `, bold: true, size: 24 }), new TextRun({ text: caption, size: 24 })] }),
    ] }),
  ];
}
function tcaption(chap, caption) {
  const n = `${chap}.${TABLES.filter((x) => x.chap === chap).length + 1}`;
  const id = newId("tab");
  TABLES.push({ chap, n, caption, id });
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 }, children: [
    new Bookmark({ id, children: [new TextRun({ text: `Table ${n}: `, bold: true, size: 24 }), new TextRun({ text: caption, size: 24 })] }),
  ] });
}
const after = () => new Paragraph({ spacing: { after: 120 }, children: [] });
const highlight = (t) => new TextRun({ text: t, highlight: "yellow" });

// ---------------------------------------------------------- front pages ---
const STUDENTS = [
  ["Pranav Ladha", "23FE10CDS00455"],
  ["Devisha Jain", "23FE10CDS00493"],
  ["Nandini Nahata", "23FE10CDS00376"],
  ["Jiya Gakhar", "23FE10CDS00480"],
];
const TITLE = "MedTwin: Realistic and Private Synthetic Patient Data with Missing Values";

function cover() {
  const L = (text, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 120, before: o.before ?? 0 },
    children: [new TextRun({ text, bold: o.bold, italics: o.italics, size: o.size || 24, font: FONT })] });
  return [
    L("A Project Report", { bold: true, size: 36, before: 400 }),
    L("on", { italics: true, size: 30 }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 200 },
      children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, size: 40, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
      new TextRun({ text: "carried out as part of the ", italics: true, size: 24 }),
      new TextRun({ text: "Project ", bold: true, italics: true, size: 28 }), new TextRun({ text: "(Subject Code)", bold: true, italics: true, size: 28, highlight: "yellow" }),
      new TextRun({ text: " Submitted", italics: true, size: 24 })] }),
    L("by", { italics: true, after: 160 }),
    ...STUDENTS.map(([n, r]) => L(`${n} (${r})`, { bold: true, size: 28, after: 40 })),
    L("in partial fulfilment for the award of the degree", { italics: true, size: 28, before: 200 }),
    L("of", { bold: true, size: 30 }),
    L("BACHELOR OF TECHNOLOGY (B.Tech)", { bold: true, size: 30 }),
    L("in", { size: 30 }),
    L("Department of Data Science and Engineering", { size: 30, after: 300 }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
      children: [image(path.join(HERE, "assets", "muj_logo.jpeg"), 250)] }),
    L("School of Computer Science and Engineering,", { bold: true, size: 30, after: 200 }),
    L("MANIPAL UNIVERSITY JAIPUR", { bold: true, size: 30, after: 40 }),
    L("RAJASTHAN, INDIA", { bold: true, size: 30, after: 300 }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "December 2026", bold: true, size: 30, highlight: "yellow" })] }),
  ];
}

function certificate() {
  const names = STUDENTS.map(([n, r]) => `${n} (${r})`);
  const who = names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
  return [
    new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.RIGHT, spacing: { before: 400, after: 400 },
      children: [new TextRun({ text: "Date: " }), highlight("__________")] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
      children: [new TextRun({ text: "CERTIFICATE", bold: true, size: 32, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 200 }, children: [
      new TextRun("This is to certify that the project entitled "),
      new TextRun({ text: `\"${TITLE}\"`, bold: true }),
      new TextRun(` has been carried out by ${who}, students of the Department of Data Science and Engineering, Manipal University Jaipur, in partial fulfilment of the requirements for the award of the degree of `),
      new TextRun({ text: "Bachelor of Technology", bold: true }),
      new TextRun(". The work presented in this report is a bonafide record of the work done by them under my guidance and supervision during "),
      highlight("[start month, year]"), new TextRun(" to "), highlight("[end month, year]"), new TextRun("."),
    ] }),
    new Paragraph({ spacing: { before: 1400 }, alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Dr. Chirag Joshi", bold: true, size: 28 })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Assistant Professor (Selection Grade)", bold: true, size: 28 })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Department of Data Science & Engineering", bold: true, size: 24 })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Manipal University Jaipur", bold: true, size: 24 })] }),
  ];
}

function acknowledgments() {
  const AK = (t) => P(t, { line: 297, run: { size: 24 } });
  return [
    unnumbered("ACKNOWLEDGMENTS", { front: true, pageBreak: false, toc: "Acknowledgement" }),
    AK("We thank our project guide, **Dr. Chirag Joshi**, Assistant Professor (Selection Grade), Department of Data Science & Engineering, Manipal University Jaipur, for guiding us through this project. He pushed us to evaluate synthetic data fairly and to test every claim we made, and the project is much better for it."),
    AK("We also thank the Head of the Department and the faculty of the Department of Data Science and Engineering for their support, and the School of Computer Science and Engineering and Manipal University Jaipur for letting us take up this work as part of our B.Tech programme."),
    AK("The UCI Heart Disease and Pima Indians Diabetes datasets are public. We are grateful to the people who collected and shared them, and to the authors of the open-source libraries we used."),
    AK("Parts of the code, analysis and text of this project were developed with the help of an AI assistant (Claude, Anthropic). All results were produced by running the project's code, and we have reviewed and verified the code, results and text and take full responsibility for them."),
    AK("Finally, we thank our families and friends for their support."),
    new Paragraph({ spacing: { before: 600 }, alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Pranav Ladha", bold: true })] }),
    ...STUDENTS.slice(1).map(([n]) => new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: n, bold: true })] })),
  ];
}

function abstract() {
  const md = (k) => hm(MT, k);
  const AP = (t) => P(t, { line: 240, after: 0 });
  return [
    unnumbered("ABSTRACT", { front: true, toc: "Abstract" }),
    AP("Medical machine-learning models need patient data, but privacy laws and hospital rules limit who may see it. Synthetic data is one answer. A generative model learns the patterns in real patient records and then produces artificial records that follow the same patterns. Real clinical data has gaps, and they are not random: whether a test was done often depends on the hospital. Most synthetic-data tools fill these gaps before training, so their output has no gaps at all. Records like that do not look like real hospital data and are easy to tell apart from it."),
    AP(`This project presents **MedTwin**, a conditional Wasserstein generative adversarial network with gradient penalty (WGAN-GP) that keeps "not measured" as part of each record. For every column that can be missing, MedTwin learns a yes/no flag that says whether the value was measured. A second flag handles values that are exactly zero, and each synthetic patient is generated for a chosen hospital, so the model picks up each hospital's testing habits. We also propose two ways to evaluate missing values. Missing-value fidelity scores check whether the gaps in synthetic data are realistic per column, per hospital and as whole patterns. A missing-value privacy audit checks whether keeping realistic gaps makes patients easier to identify.`),
    AP(`We compared MedTwin with CTGAN, TVAE and TabDDPM on the four-hospital UCI Heart Disease data (${H.n} patients) and the Pima Indians Diabetes data (${DB.n} patients). Each comparison was repeated five times and always tested on real patients that no model had seen. On heart disease, a classifier could tell MedTwin's patients from real ones ${m(md("detectability_pct"), 1)}% of the time, against ${m(hm("TabDDPM", "detectability_pct"), 1)}–${m(hm("CTGAN", "detectability_pct"), 1)}% for the other methods (50% would mean they cannot be told apart at all). A disease predictor trained only on MedTwin's data kept ${f(md("auc_ratio").mean * 100, 1)}% of the performance of one trained on real data, level with the best other method. MedTwin's missing-value patterns were almost as close to the real data as a second sample of real patients (pattern distance ${m(md("pattern_tvd"))} against ${m(md("pattern_tvd_ref"))}), and no other method put the gaps at the right hospitals. Privacy attacks found no sign that MedTwin memorised its training patients, while TabDDPM did memorise them on the diabetes data. We also found that keeping realistic gaps has a small privacy cost for patients whose gaps are unusual.`),
    new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "Keywords: ", bold: true }),
      new TextRun("synthetic data, generative adversarial networks, missing data, electronic health records, privacy evaluation, deep learning")] }),
  ];
}

const refCell = (id, width, pad) => cell([new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
  children: [new PageReference(id)] })], width, { pad });

function listTable(title, items, label, tocTitle) {
  const w = scale([1300, 6266, 1100]);
  return [
    unnumbered(title, { front: true, toc: tocTitle }),
    new Table({
      width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: w,
      rows: [
        new TableRow({ tableHeader: true, children: [cell(`${label} No`, w[0], { bold: true, fill: "D9E2F3" }),
          cell(`${label} Title`, w[1], { bold: true, fill: "D9E2F3" }), cell("Page No", w[2], { bold: true, fill: "D9E2F3", align: AlignmentType.CENTER })] }),
        ...items.map((it) => new TableRow({ cantSplit: true, children: [cell(`${label} ${it.n}`, w[0]), cell(it.caption, w[1]), refCell(it.id, w[2])] })),
      ],
    }),
  ];
}

function contentsTable() {
  const w = scale([1500, 6266, 1100]);
  const rows = CONTENTS.filter((c) => !c.skipToc).map((c) => {
    const chap = c.level === 0 && (c.label.startsWith("Chapter") || c.label === "Last Chapter");
    const fill = chap ? "F2F2F2" : undefined;
    return new TableRow({ cantSplit: true, children: [
      cell(c.label, w[0], { bold: true, fill, size: 24, pad: 4, align: c.level ? AlignmentType.CENTER : AlignmentType.LEFT }),
      cell(c.title, w[1], { bold: c.level === 0 && !c.front, fill, size: 24, pad: 4 }),
      refCell(c.id, w[2], 4),
    ] });
  });
  return [
    unnumbered("Contents", { front: true }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: w, rows: [
      new TableRow({ tableHeader: true, children: [cell("", w[0], { fill: "D9E2F3" }), cell("Title", w[1], { bold: true, fill: "D9E2F3" }),
        cell("Page No", w[2], { bold: true, fill: "D9E2F3", align: AlignmentType.CENTER })] }),
      ...rows,
    ] }),
  ];
}

function abbreviations() {
  const rows = [
    ["AI", "Artificial Intelligence"], ["AUC", "Area Under the (ROC) Curve"],
    ["CSV", "Comma-Separated Values (file format)"], ["CTGAN", "Conditional Tabular Generative Adversarial Network"],
    ["DCR", "Distance to Closest Record"], ["EHR", "Electronic Health Record"],
    ["GAN", "Generative Adversarial Network"], ["MIA", "Membership Inference Attack"],
    ["ML", "Machine Learning"], ["MUJ", "Manipal University Jaipur"],
    ["RF", "Random Forest"], ["SD", "Standard Deviation"],
    ["TabDDPM", "Tabular Denoising Diffusion Probabilistic Model"], ["TRTR", "Train on Real, Test on Real"],
    ["TSTR", "Train on Synthetic, Test on Real"], ["TVAE", "Tabular Variational Autoencoder"],
    ["TVD", "Total Variation Distance"], ["UCI", "University of California, Irvine (Machine Learning Repository)"],
    ["VA", "Veterans Affairs (VA Long Beach hospital)"], ["WGAN-GP", "Wasserstein GAN with Gradient Penalty"],
    ["YAML", "YAML Ain't Markup Language (settings file format)"],
  ];
  return [unnumbered("List of Abbreviations", { front: true }), table(["Abbreviation", "Full Form"], rows, scale([2200, 6466]))];
}

// ------------------------------------------------------------ chapter 1 ---
function chapter1() {
  const mbs = H.missing_by_site;
  return [
    ...chapter(1, "INTRODUCTION"),
    section("1.1", "Introduction to Work Done / Motivation (Overview, Applications & Advantages)"),
    sub("Overview"),
    P(`Hospitals now use machine learning to predict disease and to help doctors with diagnosis and treatment. Building these models takes a lot of real patient data. That data is sensitive, so laws such as HIPAA in the USA and GDPR in Europe, along with hospital ethics rules, limit who can see it. Students, researchers and small organisations often cannot get access, and some useful projects never start.`),
    P(`Synthetic data is one way around this ${cite("goncalves", "yan")}. A generative model learns the statistical patterns of real patient records and then produces new records that follow those patterns but belong to no real person. If the synthetic records look real, work well for training models and do not give away the real patients, they can be shared in place of the real data, for example for research, teaching or software testing.`),
    P(`In this project we built **MedTwin**, a model that creates synthetic patient records from real hospital data. MedTwin is a conditional Wasserstein generative adversarial network with gradient penalty (WGAN-GP) ${cite("goodfellow", "arjovsky", "gulrajani")}. We tested it against three widely used synthetic-data methods on two public datasets: the UCI Heart Disease data from four hospitals ${cite("janosi", "detrano")} and the Pima Indians Diabetes data ${cite("smith")}.`),
    sub("Motivation"),
    P(`Real clinical tables have gaps, and the gaps follow a pattern. Whether a test was done depends on the hospital, the doctor and the patient. Some examples from the heart disease data used here:`),
    ...bullets([
      `every patient from Switzerland has a cholesterol value of 0, which really means \"not measured\" (${f(mbs.Switzerland.chol, 0)}% missing);`,
      `the number of major blood vessels seen on a scan is missing for ${f(mbs.Switzerland.ca, 0)}–${f(mbs.Hungary.ca, 0)}% of patients at three of the four hospitals, but almost never at Cleveland;`,
      `resting blood pressure is missing for ${f(mbs["VA Long Beach"].trestbps, 0)}% of patients at VA Long Beach but for almost nobody elsewhere.`,
    ]),
    P(`Most synthetic-data tools cannot handle missing values, so people usually fill each gap with a typical value such as the median before training. The synthetic data then has no gaps at all, unlike real hospital data, and the filled-in values were never measured on anyone. A synthetic copy of a multi-hospital dataset should reproduce *which* tests were done *where*. That is the starting point of this project.`),
    sub("Applications"),
    ...bullets([
      "Hospitals could share synthetic versions of their data with researchers without exposing real patients.",
      "Students could practise building medical models on data that has real-world problems such as missing values.",
      "Developers could test data pipelines and prediction models on data that behaves like hospital data, gaps included.",
      "Because MedTwin generates patients for a chosen hospital, one model can represent the testing habits of several hospitals.",
      "The evaluation tools from this project can be used to compare any synthetic-data generator, including on how well it handles missing values.",
    ]),
    sub("Advantages"),
    ...bullets([
      "Synthetic records do not belong to real patients, so sharing them carries less privacy risk.",
      "Missing values are generated as gaps instead of being filled with made-up numbers, so the data keeps the structure of real records.",
      "A new dataset only needs a new settings file. The code stays the same.",
      "Every result is measured on real patients the model never saw, repeated with five seeds and compared with paired statistics.",
    ]),
    section("1.2", "Project Statement / Objectives of the Project"),
    P(`**Problem statement.** Build and test a model that generates synthetic patient records from incomplete clinical data collected at several hospitals, keeping the real pattern of missing values. The records should look real, be useful for training prediction models and protect the real patients. We also need ways to measure how realistic the missing values in synthetic data are, and whether keeping them puts patients at risk.`),
    P("The main objectives of the project are:"),
    ...numbered([
      "Build a generator (MedTwin) that learns which values were measured, generates each patient for a specific hospital and reproduces values that are exactly zero.",
      "Make the system work for any table of medical data through a settings file, and show this on two datasets.",
      "Test on real patients held out from training, with five repeated runs and paired statistical tests.",
      "Compare MedTwin with CTGAN, TVAE and TabDDPM under the same conditions.",
      "Define scores that measure how realistic the gaps in synthetic data are, per column, per hospital and as whole patterns.",
      "Test whether keeping realistic gaps makes patients easier to identify.",
      "Measure what each part of the model adds by switching parts off one at a time, and check how much the results depend on the main settings.",
    ]),
    section("1.3", "Organization of Report"),
    P("The rest of this report is laid out as follows."),
    ...numbered([
      "Chapter 1 introduces the problem, its applications and the objectives.",
      "Chapter 2 covers the background (synthetic data, GANs, missing data, evaluation and privacy), the tools we used and related work.",
      "Chapter 3 describes the method: datasets, data preparation, model design, training, the test protocol and the new missing-value measures, with block diagrams.",
      "Chapter 4 describes the software modules, the dashboard prototype and the automated tests.",
      "Chapter 5 presents the results on both datasets, the missing-value scores, the privacy audit, the ablation and sensitivity studies, and the problems we ran into.",
      "Chapter 6 gives the conclusions and ideas for future work.",
      "The references (IEEE style) and annexures come last. The annexures hold sample synthetic records, code snippets and instructions for running the project.",
    ]),
  ];
}

// ------------------------------------------------------------ chapter 2 ---
function chapter2() {
  const tech = [
    ["Programming language", "Python 3.9", "All code"],
    ["Deep learning", `PyTorch 2.8.0 ${cite("paszke")}`, "Generator, critic and training"],
    ["Data handling", "pandas 2.3.3, NumPy 2.0.2", "Loading, cleaning and encoding data"],
    ["Machine learning", `scikit-learn 1.6.1 ${cite("pedregosa")}`, "Random forests, nearest neighbours, AUC, scaling"],
    ["Comparison methods", "ctgan 0.12.1 (CTGAN, TVAE); TabDDPM written in this project", "Baselines"],
    ["Statistics", "SciPy", "Paired t-tests and confidence intervals"],
    ["Visualisation", "matplotlib 3.9.4, seaborn 0.13.2", "Figures"],
    ["Dashboard", "Streamlit 1.50.0", "Interactive web prototype"],
    ["Configuration", "PyYAML 6.0.3", "Dataset settings files"],
    ["Testing", "pytest", `${D.n_tests} automated tests`],
    ["Version control", "Git", "History of all changes"],
    ["Documents", "LaTeX (Springer LNCS), Tectonic", "Research paper"],
  ];
  const lit = [
    [`Wan Abu Bakar et al., 2025 ${cite("wanabubakar")}`, "CTGAN on Cleveland heart data (303 patients); gaps filled with the most common value; single run", "56% accuracy training on synthetic, testing on real (5-class)", "We use all 4 hospitals, unseen test patients and 5 repeats; MedTwin clearly beats our own CTGAN baseline"],
    [`Subedi et al., 2025 ${cite("subedi")}`, "CTGAN on Pima diabetes", "Random forest accuracy 82% (real) → 63% (synthetic)", `MedTwin keeps ${f(dm(MT, "auc_ratio").mean * 100, 0)}% of real-data performance (AUC ratio)`],
    [`Alqulaity & Yang, 2024 ${cite("alqulaity")}`, "Enhanced conditional GAN, 218 cardiovascular patients, one hospital", "Distribution statistics only; no utility or privacy test", "We test realism, usefulness and privacy on unseen patients"],
    [`Lazzaro et al., 2026 ${cite("lazzaro")}`, "Bayesian networks on UCI heart sites, trained on Cleveland only", "Data augmentation results", "We model all 4 hospitals, including the very incomplete ones"],
    [`HealthGen ${cite("healthgen")}, MissDiff ${cite("missdiff")}, Wang et al. ${cite("reinvented")}, IGAMT ${cite("igamt")}`, "Generators that keep missing values (time series, diffusion, plug-in, private EHR)", "Show that keeping missingness matters", "Same goal; we add hospital-conditioned gaps, scores for gap realism and the first audit of their privacy cost"],
  ];
  return [
    ...chapter(2, "BACKGROUND MATERIAL"),
    section("2.1", "Conceptual Overview (Concepts / Theory Used)"),
    sub("1. Synthetic data"),
    P(`Synthetic data is artificial data made by a model that has learned the patterns of real data. Reviews of the field ${cite("goncalves", "hernandez")} judge it on realism, usefulness and privacy. Realistic data looks like the real data, both column by column and in how the columns relate. Useful data trains models that work on real patients. Private data does not reveal which real people were used to make it.`),
    sub("2. Generative Adversarial Networks (GANs)"),
    P(`A GAN ${cite("goodfellow")} has two neural networks that are trained against each other. The generator turns random noise into fake records, and the discriminator tries to tell the fake records from real ones. Over training, the generator's records get harder and harder to tell apart from real data.`),
    sub("3. Wasserstein GAN with Gradient Penalty (WGAN-GP)"),
    P(`Standard GANs are often unstable and can end up producing only a few kinds of record. The Wasserstein GAN ${cite("arjovsky")} replaces the discriminator's real/fake decision with a critic that gives each record a score, and these scores estimate the Wasserstein distance between real and fake data. The gradient-penalty version ${cite("gulrajani")} penalises large gradients in the critic, which makes training much more stable. The critic loss is:`),
    C("*L*D = E[D(fake, c)] − E[D(real, c)] + λ · E[(‖∇D(mixed, c)‖ − 1)²],   with λ = 10."),
    sub("4. Conditional generation"),
    P(`In a conditional GAN, both networks also receive some extra information, called the condition. In MedTwin the condition is the hospital. The generator creates a patient *for* a given hospital, and the critic judges whether the patient is realistic *for that hospital*. This is how the model learns which tests each hospital does.`),
    sub("5. Gumbel-softmax for categories"),
    P(`Categorical columns (such as chest pain type) must take exactly one value. The Gumbel-softmax technique ${cite("jang")} lets a neural network produce a smooth, trainable approximation of a one-hot choice during training and an exact one-hot choice when sampling.`),
    sub("6. Missing data"),
    P(`Values go missing for many reasons. In data from several hospitals, the main reason is usually local practice: some hospitals just do not run a given test. Common tools either fill missing values with a typical value, which is called imputation ${cite("gain")}, or learn a separate model of which values are missing ${cite("misgan")}. If imputation is used before generating synthetic data, the gaps disappear and the synthetic records no longer show which tests were done.`),
    sub("7. Evaluating synthetic data"),
    ...bullets([
      "Fidelity measures how closely the distribution of each column matches the real data. We measure it as histogram overlap.",
      `Detectability is the accuracy of a classifier trained to tell real records from synthetic ones. At 50% the two cannot be told apart. It measures realism and says nothing about privacy ${cite("alaa")}.`,
      `TSTR (Train on Synthetic, Test on Real) ${cite("esteban")} trains a disease predictor on synthetic data and tests it on real patients. Its AUC is compared with the same predictor trained on real data (TRTR).`,
      "Distance to Closest Record (DCR) checks whether synthetic records sit suspiciously close to the patients used for training.",
      `A Membership Inference Attack (MIA) ${cite("hayes", "chen")} tries to tell whether a given real patient was in the training data. An AUC of 0.5 means the attack does no better than guessing.`,
    ]),
    P(`Passing these attacks is evidence of privacy, but it is not a guarantee ${cite("stadler")}.`),
    sub("8. Statistical testing and ablation"),
    P("Results change with random choices, so each experiment is repeated with five seeds. Methods are compared seed by seed with a paired t-based 95% confidence interval, and we call a difference *clear* only when the interval does not include zero. An ablation study switches off one part of the system at a time to see what that part adds."),
    section("2.2", "Technologies Involved"),
    P("All the code is written in Python with open-source libraries. Table 2.1 lists the main ones and what we used them for."),
    tcaption(2, "Technologies and tools used"),
    table(["Area", "Technology / version", "Used for"], tech, scale([2200, 3600, 2866])),
    after(),
    P(`Random forests ${cite("breiman")} were used for the detectability, utility and privacy tests, and the Adam optimiser ${cite("kingma")} for training the networks. The baselines CTGAN and TVAE ${cite("xu")} come from the open-source ctgan library, and TabDDPM ${cite("kotelnikov")} was re-implemented in this project following its published design.`),
    section("2.3", "Literature Survey"),
    P(`Early work on synthetic health records used GANs for binary and count features (medGAN ${cite("choi")}). CTGAN ${cite("xu")} introduced a conditional GAN for mixed tables of numbers and categories, and the same paper proposed TVAE, a variational autoencoder for tables. CTAB-GAN ${cite("zhao")} improved the handling of mixed and long-tailed variables. Diffusion models such as TabDDPM ${cite("kotelnikov")} and TabSyn ${cite("zhang")} now top many tabular benchmarks. EHR-Safe ${cite("ehrsafe")} was built for the messy structure of real electronic health records. Large benchmark studies ${cite("yan", "hernandez", "synthcity")} compare many generators on realism, usefulness and privacy, and recent work shows that tabular diffusion models can memorise their training records ${cite("fang")}.`),
    P(`Several recent studies generated synthetic versions of the same or similar data, and Table 2.2 compares them with this project. Their numbers cannot be compared directly with ours, because each study uses different data, splits and measures.`),
    tcaption(2, "Comparison with recent related work"),
    table(["Work", "What they did", "Their result", "How this project compares"], lit, scale([1900, 2350, 2000, 2416]), { size: 20 }),
    after(),
    section("2.4", "Research Gap"),
    P("The survey showed these open problems:"),
    ...numbered([
      "Studies on the UCI heart and Pima data usually fill missing values, use a single run, test on patients the generator saw, and do not measure privacy.",
      "Methods that keep missing values exist, but there is no standard way to measure whether the generated gaps are realistic, particularly per hospital.",
      "No study has tested whether keeping realistic missing values creates a privacy risk for patients with unusual gaps.",
    ]),
  ];
}

// ------------------------------------------------------------ chapter 3 ---
function chapter3() {
  const cols = Object.keys(H.missing_pct);
  const desc = { age: "Age in years", sex: "Sex", dataset: "Hospital (site)", cp: "Chest pain type", trestbps: "Resting blood pressure (mmHg)",
    chol: "Serum cholesterol (mg/dl)", fbs: "Fasting blood sugar > 120 mg/dl", restecg: "Resting ECG result", thalch: "Maximum heart rate (bpm)",
    exang: "Chest pain during exercise", oldpeak: "ST depression on ECG", slope: "Slope of ST segment", ca: "Major vessels seen on scan (0–3)",
    thal: "Thalassemia test result", num: "Disease severity (0 = none, 1–4)" };
  const colRows = cols.map((c) => [c, desc[c] || c, c === H.condition ? "site" : (H.numeric.includes(c) ? "number" : "category"), `${f(H.missing_pct[c], 1)}%`]);
  const siteCols = ["chol", "trestbps", "ca", "thal", "slope", "fbs"];
  const siteRows = Object.entries(H.missing_by_site).map(([s, v]) => [`${s} (${H.sites[s]})`, ...siteCols.map((c) => `${f(v[c], 0)}%`)]);
  const arch = [
    ["Generator input", "Random noise (128 numbers) + one-hot hospital condition"],
    ["Generator layers", "Linear 256 → LeakyReLU(0.2) → BatchNorm → Linear 512 → LeakyReLU → BatchNorm → Linear (output)"],
    ["Generator outputs", "Sigmoid for numbers and flags (0–1); Gumbel-softmax (τ = 0.2) per categorical column"],
    ["Critic layers", "Linear 512 → LeakyReLU → Dropout 0.3 → Linear 256 → LeakyReLU → Dropout 0.3 → Linear 1"],
    ["Critic input", "Record + hospital condition, so realism is judged per hospital"],
    ["Loss", "Wasserstein loss + gradient penalty (λ = 10) + correlation penalty (α = 2)"],
  ];
  const train = [
    ["Optimiser", `Adam ${cite("kingma")}, learning rate 0.0001, betas (0.0, 0.9)`],
    ["Epochs / batch size", `${D.settings.epochs} epochs, batch size 64`],
    ["Critic updates", "5 critic updates per generator update"],
    ["Records generated", `${D.settings.samples} per run`],
    ["Train / test split", "80% / 20%, stratified by outcome"],
    ["Repeats", "5 seeds (0–4); each seed changes the split and the model's starting point"],
  ];
  const measures = [
    ["Fidelity (%)", "Do single columns look real? (histogram overlap)", "Higher", "Realism"],
    ["Detectability (%)", "Can a classifier tell fake from real?", "50%", "Realism"],
    ["Correlation gap", "Do relationships between columns match?", "0", "Realism"],
    ["TSTR AUC / AUC ratio", "Does a predictor trained on fake data work on real patients?", "Higher / 1.0", "Usefulness"],
    ["DCR (%)", "Are fake records closer to training patients than to unseen ones?", "50%", "Privacy"],
    ["MIA AUC", "Can an attacker tell which patients were used for training?", "0.5", "Privacy"],
    ["Rate gap (new)", "Is each column missing as often as in real data?", "0", "Missing values"],
    ["Site gap (new)", "Is each column missing as often at each hospital?", "0", "Missing values"],
    ["Pattern distance (new)", "Do whole patterns of gaps match?", "0", "Missing values"],
    ["Rare-pattern MIA (new)", "Are patients with unusual gaps easier to identify?", "0.5", "Privacy of gaps"],
    ["Unique-pattern leak (new)", "Do one-of-a-kind gap patterns of training patients reappear?", "0", "Privacy of gaps"],
  ];
  return [
    ...chapter(3, "METHODOLOGY"),
    section("3.1", "Detailed Methodology that will be Adopted"),
    P("The project was carried out in eight stages, summarised in the block diagram of Fig. 3.1 (Section 3.2). Each stage is described below."),
    sub("Step 1: Data collection and understanding"),
    P(`We used two public datasets. The UCI Heart Disease data ${cite("janosi", "detrano")} combines patients from four hospitals, and the Pima Indians Diabetes data ${cite("smith")} was taken from OpenML (dataset 37) ${cite("openml")}. Table 3.1 summarises both.`),
    tcaption(3, "Summary of the datasets"),
    table(["", "Heart disease (UCI)", "Pima diabetes"], [
      ["Patients", H.n, DB.n],
      ["Columns used", `${H.numeric.length} numbers + ${H.categorical.length} categories + hospital`, `${DB.numeric.length} numbers`],
      ["Hospitals", Object.entries(H.sites).map(([s, n]) => `${s} (${n})`).join(", "), "1 (one clinic)"],
      ["Prediction target", `heart disease (severity > 0): ${f(H.positive_pct, 1)}% yes`, `diabetes: ${f(DB.positive_pct, 1)}% yes`],
      ["Missing values", "many; depend strongly on the hospital", "zeros that mean \"not measured\""],
    ], scale([1800, 4100, 2766])),
    after(),
    P("Table 3.2 lists the heart disease columns and how often each is missing. Table 3.3 shows that the missing values depend strongly on the hospital. A realistic synthetic dataset has to reproduce this."),
    tcaption(3, "Heart disease columns"),
    table(["Column", "Meaning", "Type", "Missing"], colRows, scale([1500, 4300, 1400, 1466])),
    after(),
    tcaption(3, "Share of heart disease patients with each value not measured, by hospital"),
    table(["Hospital (patients)", ...siteCols], siteRows, scale([2366, 1050, 1050, 1050, 1050, 1050, 1050]), { center: true }),
    after(),
    P(`For the diabetes data, a value of 0 for glucose, blood pressure, skin fold, insulin or BMI is physically impossible and means \"not measured\" (insulin is missing for ${f(DB.missing_pct.insu, 1)}% of patients). Zero pregnancies, however, is a real value. These rules are written in each dataset's settings file.`),
    sub("Step 2: Train / test split"),
    P("For each experiment, 20% of patients are held out and never shown to any model. All usefulness and privacy tests are measured against these unseen patients. The split is stratified by the outcome so both parts have the same share of sick patients."),
    sub("Step 3: Encoding records with their missing values"),
    ...bullets([
      "Zeros that mean missing (listed in the settings file) are turned into blanks.",
      "Every number column that is sometimes missing gets a 0/1 flag that says whether it was measured. The blank itself is replaced by the median, which is only a placeholder because the flag marks it as meaningless.",
      `${f(H.zero_pct.oldpeak, 0)}% of heart patients have an ST depression of exactly zero. A separate 0/1 zero flag lets the network reproduce this spike.`,
      "Numbers are scaled to the range 0–1, and each category column is one-hot encoded with \"not measured\" as an extra category.",
      "The hospital is not generated. It is given to the model as a one-hot condition.",
    ]),
    sub("Step 4: Model design and training"),
    P(`MedTwin is a conditional WGAN-GP. Its architecture is given in Table 3.4 and drawn in Fig. 3.2. In addition to the Wasserstein loss and gradient penalty, the generator is penalised when the correlations between columns in its output differ from those in the real batch (correlation loss, weight α = 2). Columns that are constant within a batch are skipped, because their correlation is undefined and would otherwise break training. Training settings are listed in Table 3.5.`),
    tcaption(3, "MedTwin model architecture"),
    table(["Part", "Specification"], arch, scale([2000, 6666])),
    after(),
    tcaption(3, "Training settings"),
    table(["Setting", "Value"], train, scale([2400, 6266])),
    after(),
    sub("Step 5: Generation and decoding"),
    P(`To create synthetic data, a hospital is drawn for each patient in the same proportions as the real data, and the generator produces ${D.settings.samples} patients. Values are converted back to real units. If a patient's zero flag is set, the value becomes 0; if its measurement flag is set, the value is left blank (\"not measured\").`),
    sub("Step 6: Cleaning"),
    P("Generated values are rounded to the precision used in hospital records (for example, ST depression to 0.1) and checked against medically possible ranges. The ranges were chosen to include every real measured value, so no real patient would be removed."),
    sub("Step 7: Evaluation protocol"),
    P(`All four methods receive the same data, the same missing-value flags, the same cleaning and the same tests. Each experiment is repeated with five seeds, and methods are compared seed by seed with paired 95% confidence intervals. Table 3.6 lists every measure used; those marked \"new\" were developed in this project.`),
    tcaption(3, "Evaluation measures"),
    table(["Measure", "Question it answers", "Best", "Group"], measures, scale([2300, 4000, 1000, 1366]), { size: 20 }),
    after(),
    sub("Step 8: New missing-value measures"),
    P("A patient's *missingness pattern* is the list of which of their values are blank, together with their hospital. We built the following measures on it."),
    ...bullets([
      "The rate gap is the average difference, over columns, in the share of blank values between real and synthetic data.",
      "The site gap is the same difference computed for each hospital and column, averaged with hospital weights. It catches a model that gets the overall rates right but puts the gaps at the wrong hospital.",
      "The pattern distance is the total variation distance between the distributions of whole patterns, ½ Σ |P_real(p) − P_synthetic(p)|. It also captures which values go missing together.",
      "These scores depend on sample size, so we score the synthetic data on samples the size of the held-out set and score the held-out real patients the same way. We call the second number the real-vs-real reference. A synthetic score close to it is as good as a second sample of real patients.",
      "The rare-pattern membership attack is the usual distance-based membership attack, run separately on patients whose pattern is rare (held by at most 2% of patients) and on everyone else.",
      "The unique-pattern leak looks at patterns held by exactly one patient. It is the share of them that shows up again in the synthetic data when that patient was in training, minus the share when the patient was held out.",
    ]),
    sub("Ablation and sensitivity design"),
    P("To measure what each component contributes, MedTwin was retrained with one component switched off at a time: missing-value flags, zero flag, hospital conditioning and correlation loss. To check that results do not depend on lucky settings, the correlation weight (0.5 and 8 instead of 2) and training length (300 and 1200 instead of 600 epochs) were varied."),
    section("3.2", "Circuit Layouts / Block Diagrams"),
    P("Fig. 3.1 shows the complete MedTwin pipeline, from the real data and settings file to the evaluated synthetic data."),
    ...figure(3, "docs/muj_report/assets/block_pipeline.png", "Block diagram of the MedTwin pipeline"),
    P("Fig. 3.2 shows the conditional WGAN-GP. The generator receives noise and the hospital; its outputs are split into sigmoid outputs (numbers and flags) and Gumbel-softmax outputs (categories). The critic receives either a generated or a real record together with the same hospital and outputs a realness score."),
    ...figure(3, "docs/muj_report/assets/block_architecture.png", "Architecture of the MedTwin conditional WGAN-GP"),
    P("Fig. 3.3 shows the evaluation framework: the held-out patients are never seen by the generator and are used for all usefulness, privacy and missing-value tests."),
    ...figure(3, "docs/muj_report/assets/block_evaluation.png", "Evaluation framework"),
  ];
}

// ------------------------------------------------------------ chapter 4 ---
function chapter4() {
  const rt = (k) => f(H.models[k].train_seconds.mean / 60, 1);
  const modules = [
    ["main.py", "Runs the full pipeline once and saves synthetic data and scores"],
    ["core/config.py", "Loads a dataset's settings file (columns, rules, ranges)"],
    ["core/ingestion.py", "Missing-value handling, flags, scaling, encoding"],
    ["core/generator.py", "Generator, critic and WGAN-GP training loop"],
    ["core/medtwin.py", "Train-and-generate steps; ablation and sensitivity switches"],
    ["core/medical_logic.py", "Medical range check and rounding"],
    ["core/evaluation.py", "Usefulness, privacy and correlation tests on held-out patients"],
    ["core/missingness.py", "New missing-value fidelity scores and privacy attacks"],
    ["research/baselines.py, tabddpm.py", "The three comparison methods"],
    ["research/multi_seed.py", "Five-seed comparison with paired statistics"],
    ["research/ablation.py", "Ablation and sensitivity studies"],
    ["research/visualizer.py, paper_tables.py", "Figures and paper tables"],
    ["app/dashboard.py", "Interactive Streamlit dashboard"],
    ["configs/*.yaml", "One settings file per dataset"],
    ["tests/", `${D.n_tests} automated tests`],
  ];
  const methods = [
    [`CTGAN ${cite("xu")}`, "Conditional tabular GAN; 300 epochs, batch 100", `${rt("CTGAN")} min`],
    [`TVAE ${cite("xu")}`, "Variational autoencoder for tables; 300 epochs, batch 100", `${rt("TVAE")} min`],
    [`TabDDPM ${cite("kotelnikov")}`, "Diffusion model for tables (our re-implementation); 8000 steps", `${rt("TabDDPM")} min`],
    ["MedTwin (ours)", `Conditional WGAN-GP with missing-value flags; ${D.settings.epochs} epochs`, `${rt(MT)} min`],
  ];
  return [
    ...chapter(4, "IMPLEMENTATION"),
    section("4.1", "Modules"),
    P("The system is organised into small, separately tested modules. Table 4.1 lists the main files and their purpose."),
    tcaption(4, "Software modules"),
    table(["File / folder", "Purpose"], modules, scale([3300, 5366])),
    after(),
    sub("1. Settings module"),
    P("Everything specific to a dataset lives in a YAML settings file: which columns are numbers or categories, which column is the hospital, which zeros mean \"not measured\", which columns have a spike at zero, the medically possible ranges and the prediction target. Adding a dataset only needs a new settings file and no code changes. That is how we added the diabetes data."),
    sub("2. Data ingestion module"),
    P("This module marks zero-coded values as missing, creates the missing-value and zero flags, fills blanks with placeholder medians, scales numbers to 0–1, one-hot encodes categories (with \"not measured\" as a category) and prepares the hospital condition. It also records the layout of the generator's outputs so that the model can be built automatically for any dataset."),
    sub("3. Generator and training module"),
    P("This module defines the conditional generator and critic (Table 3.4) and the WGAN-GP training loop, with five critic updates per generator update, the gradient penalty and the correlation loss. During development we found a bug here. A batch with a constant column produced NaN values, and these broke the whole model. Skipping constant columns in the correlation loss fixed it."),
    sub("4. Generation, cleaning and evaluation modules"),
    P("After training, synthetic patients are generated for hospitals drawn in real proportions, decoded back to real units, blanked where the flags say \"not measured\", range-checked and rounded. The evaluation module then computes fidelity, detectability, TSTR utility, DCR, membership inference and the correlation gap against the held-out patients, and the missingness module computes the new missing-value scores and privacy attacks."),
    sub("5. Baselines and experiment modules"),
    P("The comparison methods went through exactly the same pipeline (Table 4.2). None of them accepts missing values, so they got the same flag-and-placeholder encoding as MedTwin. The experiment scripts repeat every comparison over five seeds, save each run's results and synthetic data, and can pick up again if stopped. After we added new measurements and re-ran everything, all 400 saved results came out exactly the same, so the experiments are reproducible."),
    tcaption(4, "Methods compared and average training time (heart disease)"),
    table(["Method", "Description", "Training time"], methods, scale([2200, 4900, 1566])),
    after(),
    section("4.2", "Prototype"),
    P("A working prototype was built as an interactive web dashboard using Streamlit (Fig. 4.1). It loads the latest synthetic data and shows: the number of synthetic patients, statistical fidelity, an average clinical value and the detectability score; the held-out evaluation (TSTR utility, utility ratio, DCR and membership-inference AUC); interactive distribution and correlation comparisons between real and synthetic patients; and a data explorer in which blank values are shown as \"not measured\". Users can filter the synthetic patients by sex and age range and export the filtered records as a CSV file."),
    ...figure(4, "docs/muj_report/assets/dashboard.png", "MedTwin interactive dashboard (heart disease data)", 430),
    P("The pipeline can also be run from the command line: one command trains MedTwin, generates synthetic patients, runs all tests and prints a summary of fidelity, detectability, utility, privacy and missing-value scores. A single option switches to another dataset (Annexure C)."),
    section("4.3", "Testing"),
    P(`The project includes ${D.n_tests} automated tests (pytest) that run in under a minute with very short training. They check, among other things, that: the training matrix has no blanks and matches the generator's output layout; every generated category is exactly one value; conditioning on a hospital produces patients only for that hospital; all generated values respect the medical ranges; real patients handed in as \"synthetic\" are caught by the privacy tests; data with every gap filled gets a bad missing-value score; every ablation variant trains and produces a valid report; and results are unchanged when synthetic data is saved to CSV and read back.`),
  ];
}

// ------------------------------------------------------------ chapter 5 ---
function chapter5() {
  const models = [MT, "TVAE", "TabDDPM", "CTGAN"];
  const label = (k) => (k === MT ? "MedTwin (ours)" : k);
  const benchRows = (ds) => models.map((k) => [label(k), ms(ds.models[k].fidelity_pct, 1), ms(ds.models[k].detectability_pct, 1),
    ms(ds.models[k].tstr_auc), ms(ds.models[k].auc_ratio), ms(ds.models[k].dcr_closer_to_train_pct, 1), ms(ds.models[k].mia_auc)]);
  const benchHead = ["Method", "Fidelity % (↑)", "Detect. % (→50)", "TSTR AUC (↑)", "AUC ratio (→1)", "DCR % (→50)", "MIA AUC (→0.5)"];
  const bw = scale([1500, 1200, 1200, 1200, 1200, 1150, 1216]);
  const missRows = (ds, hasSite) => [
    ["Two real samples (reference)", m(ds.models[MT].miss_rate_gap_ref), hasSite ? m(ds.models[MT].site_miss_gap_ref) : "–", m(ds.models[MT].pattern_tvd_ref)],
    ...models.map((k) => [label(k), ms(ds.models[k].miss_rate_gap), hasSite ? ms(ds.models[k].site_miss_gap) : "–", ms(ds.models[k].pattern_tvd)]),
    ["MedTwin without missing-value flags", ms(ds.variants.no_missing_flags.miss_rate_gap), hasSite ? ms(ds.variants.no_missing_flags.site_miss_gap) : "–", ms(ds.variants.no_missing_flags.pattern_tvd)],
    ...(hasSite ? [["MedTwin without hospital conditioning", ms(ds.variants.no_condition.miss_rate_gap), ms(ds.variants.no_condition.site_miss_gap), ms(ds.variants.no_condition.pattern_tvd)]] : []),
  ];
  const privRows = [
    ["Full MedTwin", ms(hv("full", "rare_pattern_dist_mia_auc")), ms(hv("full", "common_pattern_dist_mia_auc")), ms(hv("full", "mia_auc")), ms(hv("full", "unique_pattern_leak_pct"), 1)],
    ["MedTwin without missing-value flags", ms(hv("no_missing_flags", "rare_pattern_dist_mia_auc")), ms(hv("no_missing_flags", "common_pattern_dist_mia_auc")), ms(hv("no_missing_flags", "mia_auc")), ms(hv("no_missing_flags", "unique_pattern_leak_pct"), 1)],
  ];
  const ablNames = { full: "Full MedTwin", no_missing_flags: "without missing-value flags", no_zero_flags: "without zero flag",
    no_condition: "without hospital conditioning", no_corr_loss: "without correlation loss" };
  const star = (ds, v, k) => (v !== "full" && ds.vs_full[v] && ds.vs_full[v][k] && ds.vs_full[v][k].clear ? " *" : "");
  const ablRows = (ds, list) => list.filter((v) => ds.variants[v]).map((v) => [ablNames[v],
    ms(ds.variants[v].detectability_pct, 1) + star(ds, v, "detectability_pct"),
    ms(ds.variants[v].corr_gap_numeric) + star(ds, v, "corr_gap_numeric"),
    ms(ds.variants[v].auc_ratio) + star(ds, v, "auc_ratio"),
    ms(ds.variants[v].mia_auc) + star(ds, v, "mia_auc")]);
  const sensNames = { full: "Default (weight 2, 600 epochs)", "corr_weight_0.5": "Correlation weight 0.5", corr_weight_8: "Correlation weight 8",
    epochs_300: "300 epochs", epochs_1200: "1200 epochs" };
  const sensRows = Object.keys(sensNames).map((v) => [sensNames[v], ms(hv(v, "fidelity_pct"), 1) + star(H, v, "fidelity_pct"),
    ms(hv(v, "detectability_pct"), 1) + star(H, v, "detectability_pct"), ms(hv(v, "auc_ratio")) + star(H, v, "auc_ratio"),
    ms(hv(v, "dcr_closer_to_train_pct"), 1) + star(H, v, "dcr_closer_to_train_pct"), ms(hv(v, "mia_auc")) + star(H, v, "mia_auc")]);
  const vb = (b, k) => H.vs_baselines[b][k];
  return [
    ...chapter(5, "Results and Analysis"),
    P("All results are averages over five repeated experiments ± standard deviation, measured on real patients that the models never saw. Arrows show what is better: ↑ higher, ↓ lower, → closer to the given value. We call a difference *clear* when its paired 95% confidence interval excludes zero."),
    section("5.1", "Heart Disease: MedTwin Compared with Other Methods"),
    tcaption(5, "Heart disease results (mean ± SD over five seeds)"),
    table(benchHead, benchRows(H), bw, { center: true, size: 20 }),
    after(),
    ...figure(5, "figures/heart_disease/fig_benchmark.png", "Heart disease comparison of the four methods (mean ± SD over five seeds; dashed lines show the ideal value)"),
    P(`MedTwin has the highest fidelity (${m(hm(MT, "fidelity_pct"), 1)}%) and the lowest detectability (${m(hm(MT, "detectability_pct"), 1)}%). The other methods reach ${m(hm("TabDDPM", "detectability_pct"), 1)}% (TabDDPM), ${m(hm("TVAE", "detectability_pct"), 1)}% (TVAE) and ${m(hm("CTGAN", "detectability_pct"), 1)}% (CTGAN), and both differences are clear against each of them. MedTwin's AUC ratio is ${m(hm(MT, "auc_ratio"))}, level with TVAE (${m(hm("TVAE", "auc_ratio"))}) and clearly better than CTGAN (${m(hm("CTGAN", "auc_ratio"))}). DCR and membership-inference AUC are close to their ideal values for all four methods, and none of them produced an exact copy of a training patient.`),
    P(`Generating each patient for a hospital, and generating the gaps instead of filling them, makes MedTwin's records much harder for a classifier to pick out, and the data stays just as useful. Detectability (${m(hm(MT, "detectability_pct"), 1)}%) is still above the ideal 50%, so there is room to improve.`),
    section("5.2", "Diabetes: The Same Picture"),
    tcaption(5, "Pima diabetes results (mean ± SD over five seeds)"),
    table(benchHead, benchRows(DB), bw, { center: true, size: 20 }),
    after(),
    ...bullets([
      `MedTwin again has the best fidelity (${m(dm(MT, "fidelity_pct"), 1)}%) and ties for usefulness (AUC ratio ${m(dm(MT, "auc_ratio"))}).`,
      `TabDDPM is the hardest to detect (${m(dm("TabDDPM", "detectability_pct"), 1)}%), but it memorised its training patients. ${m(dm("TabDDPM", "dcr_closer_to_train_pct"), 1)}% of its records are closer to training patients than to unseen ones, and the membership attack reaches an AUC of ${m(dm("TabDDPM", "mia_auc"))}. MedTwin shows no such leak (${m(dm(MT, "mia_auc"))}). This agrees with recent findings that tabular diffusion models memorise ${cite("fang")}, and it is a good example of why detectability should not be read as privacy.`,
    ]),
    ...figure(5, "figures/diabetes/fig_benchmark.png", "Pima diabetes comparison of the four methods"),
    section("5.3", "Are the Missing Values Realistic?"),
    tcaption(5, "Missing-value fidelity (mean ± SD; lower is better)"),
    table(["Data", "Rate gap", "Site gap", "Pattern distance"], [
      Object.assign(["Heart disease", "", "", ""], { _fill: "F2F2F2" }), ...missRows(H, true),
      Object.assign(["Diabetes (one clinic, so no site gap)", "", "", ""], { _fill: "F2F2F2" }), ...missRows(DB, false),
    ], scale([3266, 1800, 1800, 1800]), { center: true, size: 20 }),
    after(),
    ...bullets([
      `MedTwin's pattern distance on heart disease (${m(hm(MT, "pattern_tvd"))}) is close to the gap between two real samples (${m(hm(MT, "pattern_tvd_ref"))}) and clearly better than every other method in all five seeds (TVAE ${m(hm("TVAE", "pattern_tvd"))}, TabDDPM ${m(hm("TabDDPM", "pattern_tvd"))}, CTGAN ${m(hm("CTGAN", "pattern_tvd"))}).`,
      `Only MedTwin puts the gaps at the right hospitals. Its site gap (${m(hm(MT, "site_miss_gap"))}) matches the real data (${m(hm(MT, "site_miss_gap_ref"))}). The other methods got the same flags and their overall missing rates are close to real, but they do not know which hospital a gap belongs to, so they put gaps in the wrong places (site gaps ${m(hm("TVAE", "site_miss_gap"))}–${m(hm("CTGAN", "site_miss_gap"))}). A per-column check alone would not show this.`,
      `Without the missing-value flags the pattern distance jumps to ${m(hv("no_missing_flags", "pattern_tvd"))}; without hospital conditioning the overall rates stay right but the site gap (${m(hv("no_condition", "site_miss_gap"))}) and pattern distance (${m(hv("no_condition", "pattern_tvd"))}) clearly worsen.`,
      `On diabetes, MedTwin again has the lowest pattern distance (${m(dm(MT, "pattern_tvd"))}; other methods ${m(dm("TabDDPM", "pattern_tvd"))}–${m(dm("CTGAN", "pattern_tvd"))}).`,
    ]),
    ...figure(5, "figures/heart_disease/fig_missingness_by_site.png", "Share of heart disease patients with each value not measured, per hospital: real (blue) and MedTwin (orange)"),
    P("Fig. 5.3 shows that MedTwin reproduces the hospital-specific structure: cholesterol missing for every Swiss patient, vessel counts missing almost everywhere except Cleveland, and thalassemia missing for most Hungarian and VA patients. The largest differences are for fasting blood sugar in Switzerland (61% real, 72% synthetic) and ST slope in Switzerland (14% real, 7% synthetic)."),
    section("5.4", "Does Keeping Missing Values Cost Privacy?"),
    tcaption(5, "Missing-value privacy audit, heart disease (mean ± SD; 0.5 or 0 is ideal)"),
    table(["Variant", "MIA rare patterns", "MIA common patterns", "MIA all patients", "Unique leak (points)"], privRows,
      scale([2666, 1500, 1500, 1500, 1500]), { center: true, size: 20 }),
    after(),
    P(`On heart disease about ${f(hm(MT, "n_rare_patients").mean, 0)} of the 368 attacked patients per test have a rare missingness pattern. For them, the membership attack reaches an AUC of ${m(hv("full", "rare_pattern_dist_mia_auc"))} against MedTwin, compared with ${m(hv("no_missing_flags", "rare_pattern_dist_mia_auc"))} when MedTwin is trained without the missing-value flags, and it is higher in all five seeds. For patients with common patterns, and over all patients, the difference is not clear. Patterns unique to one training patient do not reappear more often than those of unseen patients, so the model is not copying patterns.`),
    P(`So keeping realistic gaps makes the data more realistic and leaves overall privacy where it was, but patients with *unusual* gaps become slightly easier to single out. The effect is small. Still, these are the patients a data owner would worry about most. On diabetes only about ${f(dm(MT, "n_rare_patients").mean, 0)} patients per test have rare patterns, which is too few to draw a conclusion.`),
    section("5.5", "What Each Component Contributes (Ablation)"),
    tcaption(5, "Ablation study (mean ± SD; * = clear difference from the full model)"),
    table(["Variant", "Detect. % (→50)", "Corr. gap (↓)", "AUC ratio (→1)", "MIA AUC (→0.5)"], [
      Object.assign(["Heart disease", "", "", "", ""], { _fill: "F2F2F2" }), ...ablRows(H, ["full", "no_missing_flags", "no_zero_flags", "no_condition", "no_corr_loss"]),
      Object.assign(["Diabetes", "", "", "", ""], { _fill: "F2F2F2" }), ...ablRows(DB, ["full", "no_missing_flags", "no_corr_loss"]),
    ], scale([2866, 1450, 1450, 1450, 1450]), { center: true, size: 20 }),
    after(),
    ...bullets([
      `The missing-value flags matter most. Without them, detectability rises from ${m(hv("full", "detectability_pct"), 1)}% to ${m(hv("no_missing_flags", "detectability_pct"), 1)}% (heart) and from ${m(dv("full", "detectability_pct"), 1)}% to ${m(dv("no_missing_flags", "detectability_pct"), 1)}% (diabetes). Filling gaps with typical values makes synthetic patients easy to spot, which matches the \"impute-then-generate\" bias reported for MissDiff ${cite("missdiff")}.`,
      `The zero flag and hospital conditioning also clearly help realism (detectability ${m(hv("no_zero_flags", "detectability_pct"), 1)}% and ${m(hv("no_condition", "detectability_pct"), 1)}% without them).`,
      `The correlation loss does not change realism or usefulness. Without it, though, the correlation gap between numeric columns nearly doubles (${m(hv("full", "corr_gap_numeric"))} → ${m(hv("no_corr_loss", "corr_gap_numeric"))} heart; ${m(dv("full", "corr_gap_numeric"))} → ${m(dv("no_corr_loss", "corr_gap_numeric"))} diabetes).`,
      "No component changes usefulness or overall privacy. The medical range check never removed a patient, so it is only a safety net.",
    ]),
    section("5.6", "Sensitivity to Settings"),
    tcaption(5, "Sensitivity of MedTwin to its main settings, heart disease (* = clear difference from the default)"),
    table(["Setting", "Fidelity %", "Detect. %", "AUC ratio", "DCR %", "MIA AUC"], sensRows, scale([2466, 1240, 1240, 1240, 1240, 1240]), { center: true, size: 20 }),
    after(),
    P(`Changing the correlation weight or training for 300 epochs makes no clear difference. Training twice as long (1200 epochs) gives the same realism and usefulness but starts to memorise: DCR rises from ${m(hv("full", "dcr_closer_to_train_pct"), 1)}% to ${m(hv("epochs_1200", "dcr_closer_to_train_pct"), 1)}% and MIA AUC from ${m(hv("full", "mia_auc"))} to ${m(hv("epochs_1200", "mia_auc"))}. So the results do not rest on lucky settings, and 600 epochs is a sensible place to stop.`),
    section("5.7", "Quality of the Generated Values"),
    P("Figs. 5.4 and 5.5 compare the distributions of the number columns, and the correlations between them, for real and synthetic heart disease patients. The shapes and the main relationships, such as maximum heart rate falling with age, are kept."),
    ...figure(5, "figures/heart_disease/fig_continuous.png", "Distributions of the number columns: real (blue) vs MedTwin (orange), measured values only"),
    ...figure(5, "figures/heart_disease/fig_correlation.png", "Correlations between number columns: real, MedTwin and their difference"),
    section("5.8", "Challenges Faced and Learnings"),
    ...bullets([
      "Training sometimes broke down. Batches that happened to contain a constant column produced NaN values in the correlation loss, and these corrupted the whole model. Skipping constant columns fixed it.",
      "Some missing values were hidden as zeros, for example Swiss cholesterol. They distorted our early results until the settings file marked them as missing.",
      "Early versions tested on patients the model had already seen, and reported a \"privacy\" score that really measured realism. Holding out patients and keeping detectability separate from privacy fixed both problems.",
      "Single runs gave misleading differences between methods. Repeating each run with five seeds and using paired confidence intervals showed which differences were real.",
      "The full comparison takes about an hour. Because each run's results and synthetic data were saved, experiments could resume after a stop, and new measures could be added without retraining.",
    ]),
  ];
}

// ------------------------------------------------------------ chapter 6 ---
function chapter6() {
  const future = [
    ["Comparison with missing-value-aware generators",
      "Compare MedTwin directly with the closest previous methods, such as MissDiff and the missing-pattern-preserving approach of Wang et al.",
      ["Benchmark all methods with the new missing-value scores.", "Identify which design choices matter most for realistic gaps."],
      "Shows how MedTwin compares with the newest methods."],
    ["Protecting patients with unusual gaps",
      "Reduce the privacy cost found for patients with rare missingness patterns.",
      ["Merge or coarsen rare patterns before release.", "Train with differential privacy and measure the realism it costs."],
      "Lets realistic gaps be kept while protecting the patients most at risk."],
    ["Generating data for new hospitals",
      "Use hospital conditioning to generate patients for a hospital the model has never seen.",
      ["Train on three hospitals and generate for the fourth.", "Test whether models trained on these patients work at the new hospital."],
      "Would help hospitals share data and build models together."],
    ["Larger and time-based records",
      "Extend MedTwin to large hospital databases and records collected over time.",
      ["Patients with many visits and changing measurements.", "Thousands of patients and many more columns."],
      "Brings the work closer to real hospital databases."],
  ];
  const out = [];
  future.forEach(([title, obj, apps, sig], i) => {
    out.push(sub(`${i + 1}. ${title}`));
    out.push(...bullets([`**Objective:** ${obj}`]));
    out.push(...bullets(["**Applications:**"]));
    out.push(...bullets(apps, 1));
    out.push(...bullets([`**Significance:** ${sig}`]));
  });
  return [
    pageSection("6.1", "Conclusions", { label: "Last Chapter", title: "CONCLUSIONS & FUTURE SCOPE" }),
    P("We built and tested MedTwin, a model that generates synthetic patient records and keeps the real pattern of missing values. Our main conclusions are below."),
    sub("1. Realistic synthetic patients, including their gaps"),
    ...bullets([
      `**Outcome:** MedTwin produced the most realistic records of the four methods tested (detectability ${m(hm(MT, "detectability_pct"), 1)}% on heart disease against ${m(hm("TabDDPM", "detectability_pct"), 1)}–${m(hm("CTGAN", "detectability_pct"), 1)}%) and the highest fidelity on both datasets.`,
      "**Significance:** keeping \"not measured\" in the data instead of filling it in is the main reason the records look real. In the ablation study, the missing-value flags made the biggest difference.",
    ]),
    sub("2. Useful and private"),
    ...bullets([
      `**Outcome:** a predictor trained on MedTwin's data keeps ${f(hm(MT, "auc_ratio").mean * 100, 1)}% (heart) and ${f(dm(MT, "auc_ratio").mean * 100, 1)}% (diabetes) of real-data performance, and privacy attacks found no memorisation.`,
      "**Significance:** MedTwin became more realistic without losing usefulness or privacy. TabDDPM, by contrast, memorised its training patients on the diabetes data.",
    ]),
    sub("3. New ways to measure missing values"),
    ...bullets([
      `**Outcome:** the new missing-value scores showed that MedTwin's gap patterns are almost as close to the real data as a second real sample (${m(hm(MT, "pattern_tvd"))} vs ${m(hm(MT, "pattern_tvd_ref"))}), and that only MedTwin places gaps at the right hospitals.`,
      "**Significance:** these scores show differences that per-column checks miss, and they work for any synthetic-data generator.",
    ]),
    sub("4. A privacy cost of keeping gaps"),
    ...bullets([
      `**Outcome:** keeping realistic gaps raises the membership attack on patients with rare patterns from ${m(hv("no_missing_flags", "rare_pattern_dist_mia_auc"))} to ${m(hv("full", "rare_pattern_dist_mia_auc"))}, while overall privacy is unchanged.`,
      "**Significance:** anyone releasing such data should know about this small cost for patients with unusual gaps. Reducing it is an obvious next step.",
    ]),
    P(`The work has limits. Both datasets are small (under 1000 patients), and detectability is still above the ideal 50%. Privacy was tested with attacks and is not guaranteed mathematically ${cite("stadler")}. The audit of gaps needs enough patients with rare patterns, so it was inconclusive on diabetes. And the newest missing-value-aware generators were not part of the comparison.`),
    pageSection("6.2", "Future Scope of Work"),
    P("Several directions follow from this work."),
    ...out,
  ];
}

// --------------------------------------------------- references, annex ---
function references() {
  const id = newId("ref");
  CONTENTS.push({ level: 0, label: "", title: "REFERENCES", id });
  return [
    new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, pageBreakBefore: true, spacing: { after: 300 },
      children: [new Bookmark({ id, children: [new TextRun({ text: "References" })] })] }),
    ...ORDER.map((k, i) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: 240 },
      indent: { left: 567, hanging: 567 }, tabStops: [{ type: TabStopType.LEFT, position: 567 }],
      children: [new TextRun({ text: `[${i + 1}]\t`, size: 24 }), new TextRun({ text: REFS[k], size: 24 })] })),
  ];
}

function annexures() {
  const id = newId("ann");
  CONTENTS.push({ level: 0, label: "", title: "ANNEXURES", id });
  const code = (lines) => lines.map((l, i) => new Paragraph({ spacing: { after: 0, line: 240 }, indent: { left: 284 },
    keepNext: i < lines.length - 1, keepLines: true,
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
    children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })] }));
  const sample = H.sample;
  const show = ["age", "gender", "chest_pain", "trestbps", "chol", "thalch", "oldpeak", "ca", "thal", "num", "origin_study"];
  const idx = show.map((c) => sample.columns.indexOf(c));
  const rows = sample.rows.map((r) => idx.map((i) => (r[i] === null ? "not measured" : String(r[i]))));
  const log = [
    ["2026-09-29", "Generated all 15 columns with \"not measured\" handling and hospital conditioning"],
    ["2026-09-29", "Added the oldpeak zero flag, 5-seed comparison and paper figures"],
    ["2026-09-29", "Moved dataset settings into configuration files; added the Pima diabetes dataset"],
    ["2026-10-15", "Added ablation switches and the ablation and sensitivity studies"],
    ["2026-10-15", "Separated detectability from privacy; added the correlation-gap measure"],
    ["2026-10-15", "Added the missing-value fidelity scores and privacy audit"],
    ["2026-10-16", "Verified all references; wrote and compiled the research paper"],
  ];
  return [
    new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, pageBreakBefore: true, spacing: { after: 300 },
      children: [new Bookmark({ id, children: [new TextRun({ text: "Annexures" })] })] }),
    sub("Annexure A: Sample synthetic patients"),
    P("The first eight synthetic heart disease patients generated by MedTwin (seed 0). Blank values are shown as \"not measured\". None of these records belongs to a real patient."),
    tcaption("A", "Sample of MedTwin's synthetic heart disease patients"),
    table(show, rows, scale(show.map(() => 1)), { size: 15 }),
    after(),
    sub("Annexure B: Code snippets"),
    P("**B.1 Creating the missing-value and zero flags** (core/ingestion.py):", { keepNext: true, align: AlignmentType.LEFT }),
    ...code([
      "flags = pd.DataFrame(",
      "    {missing_flag(c): df[c].isna().astype(float) for c in self.flagged_cols},",
      "    index=df.index,",
      ")",
      "for col in self.zero_cols:",
      "    flags[zero_flag(col)] = (df[col] == 0).astype(float)",
    ]),
    after(),
    P("**B.2 Generator output layer** (core/generator.py): sigmoid for numbers and flags, Gumbel-softmax for each category.", { keepNext: true, align: AlignmentType.LEFT }),
    ...code([
      "def forward(self, z, cond, hard=False):",
      "    logits = self.model(torch.cat([z, cond], dim=1))",
      "    outputs, start = [], 0",
      "    for kind, width in self.output_info:",
      "        segment = logits[:, start:start + width]",
      "        if kind == \"sigmoid\":",
      "            outputs.append(torch.sigmoid(segment))",
      "        else:",
      "            outputs.append(F.gumbel_softmax(segment, tau=GUMBEL_TAU, hard=hard))",
      "        start += width",
      "    return torch.cat(outputs, dim=1)",
    ]),
    after(),
    P("**B.3 Pattern distance between real and synthetic gaps** (core/missingness.py):", { keepNext: true, align: AlignmentType.LEFT }),
    ...code([
      "p_ref = patterns(reference, cols).value_counts(normalize=True)",
      "p_oth = patterns(other, cols).value_counts(normalize=True)",
      "union = p_ref.index.union(p_oth.index)",
      "scores[\"pattern_tvd\"] = float(",
      "    0.5 * (p_ref.reindex(union, fill_value=0)",
      "           - p_oth.reindex(union, fill_value=0)).abs().sum()",
      ")",
    ]),
    after(),
    sub("Annexure C: How to run the project"),
    table(["Task", "Command"], [
      ["Install", "pip install -r requirements.txt"],
      ["Run the pipeline (heart disease)", "python main.py"],
      ["Run on diabetes", "python main.py --config configs/diabetes.yaml"],
      ["Dashboard", "streamlit run app/dashboard.py"],
      ["5-seed comparison", "python research/multi_seed.py"],
      ["Ablation study", "python research/ablation.py"],
      ["Sensitivity study", "python research/ablation.py --study sensitivity"],
      ["Figures", "python research/visualizer.py --benchmark"],
      ["Automated tests", "python -m pytest tests -q"],
    ], scale([3000, 5666]), { size: 20 }),
    after(),
    sub("Annexure D: Project timeline"),
    P("An early version of the project (April 2026) used 9 columns, filled missing values and tested on patients the model had seen. It was rebuilt to handle all columns with missing values and hospitals, to test on unseen patients and to compare against other methods. The main recorded steps since then are:"),
    table(["Date", "Step"], log, scale([1700, 6966]), { size: 20 }),
  ];
}

// --------------------------------------------------------------- build ---
function build() {
  // Chapters first: they fill CONTENTS, FIGURES, TABLES and citation order
  const ack = acknowledgments();
  const abs = abstract();
  const chapters = [...chapter1(), ...chapter2(), ...chapter3(), ...chapter4(), ...chapter5(), ...chapter6()];
  const refs = references();
  const ann = annexures();

  // Front-matter entries appear before the lists, chapters after
  const front = [...listTable("LIST OF FIGURES", FIGURES, "Figure", "List Of Figures"), ...listTable("LIST OF TABLES", TABLES, "Table", "List Of Tables")];
  const abbr = abbreviations();
  // Order contents: front matter first (ack, abstract, lists, abbreviations), then chapters
  const isFront = (c) => c.front;
  const frontEntries = CONTENTS.filter(isFront);
  const rest = CONTENTS.filter((c) => !isFront(c));
  CONTENTS.length = 0;
  CONTENTS.push(...frontEntries.filter((c) => c.title !== "Contents"), ...rest);
  const tocFinal = contentsTable();
  CONTENTS.length = 0;

  const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 22 })] })] });
  const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 } };

  const doc = new Document({
    creator: "Pranav Ladha", title: TITLE, description: "MedTwin B.Tech project report",
    styles: {
      default: { document: { run: { font: FONT, size: 28 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: FONT }, paragraph: { spacing: { before: 0, after: 240 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
      ],
    },
    numbering: { config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 844, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1564, hanging: 360 } } } } ] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 844, hanging: 360 } } } } ] },
    ] },
    sections: [
      { properties: { page }, children: [...cover(), ...certificate()] },
      { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } }, footers: { default: footer },
        children: [...ack, ...abs, ...front, ...tocFinal, ...abbr] },
      { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } }, footers: { default: footer },
        children: [...chapters, ...refs, ...ann] },
    ],
  });
  return Packer.toBuffer(doc).then((buf) => {
    fs.writeFileSync(OUT, buf);
    console.log(`wrote ${OUT}: ${FIGURES.length} figures, ${TABLES.length} tables, ${ORDER.length} references`);
    const unused = Object.keys(REFS).filter((k) => !ORDER.includes(k));
    if (unused.length) console.log("references never cited:", unused.join(", "));
  });
}

build().catch((e) => { console.error(e); process.exit(1); });
