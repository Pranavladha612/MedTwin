// Builds a separate Word file with a short summary of each of the 15
// references in the MUJ report, numbered as in the report's reference list.
//
//   node docs/muj_report/build_ref_summaries.js

const fs = require("fs");
const path = require("path");
const { AlignmentType, Document, Footer, HeadingLevel, Packer, PageNumber, Paragraph, TextRun } = require("docx");

const OUT = process.env.REF_OUT || path.join(__dirname, "..", "MedTwin_Reference_Summaries.docx");
const FONT = "Times New Roman";

// [number in report, citation, what the paper did, how this project uses it]
const REFS = [
  [1, 'A. Goncalves, P. Ray, B. Soper, J. Stevens, L. Coyle and A. P. Sales, "Generation and evaluation of synthetic patient data," BMC Medical Research Methodology, vol. 20, art. no. 108, 2020.',
    "The authors trained several generators on real cancer-registry patient records. They included simple statistical models, Bayesian networks and a GAN. They then compared the synthetic data on two things: how well it kept the statistics of the real data, and how much it revealed about the real patients. The GAN did not come out on top. Some of the simpler probabilistic models gave data that was just as useful and no less private.",
    "We use this paper for the idea that synthetic patient data has to be judged on realism, usefulness and privacy together. That idea shapes our whole evaluation."],
  [2, 'I. J. Goodfellow et al., "Generative adversarial nets," Advances in Neural Information Processing Systems 27, 2014.',
    "This is the paper that introduced GANs. A generator network turns random noise into fake samples, and a discriminator network learns to tell fake samples from real ones. The two are trained against each other. The paper shows that, in theory, the best generator reproduces the real data distribution exactly.",
    "MedTwin is a GAN, so this paper is the starting point for our model."],
  [3, 'M. Arjovsky, S. Chintala and L. Bottou, "Wasserstein generative adversarial networks," Proc. 34th International Conference on Machine Learning, PMLR vol. 70, 2017.',
    "Plain GANs are hard to train, and they often collapse to a few kinds of output. This paper replaces the discriminator's real/fake decision with a critic that estimates the Wasserstein (earth mover's) distance between real and fake data. To keep the critic well behaved, the authors clip its weights. Training became more stable, and the loss now tracked sample quality.",
    "MedTwin uses the Wasserstein loss from this paper."],
  [4, 'I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin and A. Courville, "Improved training of Wasserstein GANs," Advances in Neural Information Processing Systems 30, 2017.',
    "Weight clipping in the original Wasserstein GAN caused its own training problems. This paper drops it and adds a gradient penalty instead. The critic's gradient is pushed towards a norm of 1 at points between real and fake samples. With this WGAN-GP method, many different network designs trained reliably. The paper uses a penalty weight of 10.",
    "MedTwin is a conditional WGAN-GP, with the same gradient penalty and the same weight (lambda = 10)."],
  [5, 'A. Janosi, W. Steinbrunn, M. Pfisterer and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1989.',
    "This is the public heart disease dataset. It has 920 patients from four hospitals: the Cleveland Clinic, the Hungarian Institute of Cardiology in Budapest, the university hospitals of Zurich and Basel in Switzerland, and the VA Long Beach hospital. The outcome is whether an angiogram showed heart disease. Many values are missing, and which ones are missing depends on the hospital.",
    "This is our main dataset. Its hospital-dependent gaps are the reason the project exists."],
  [6, 'J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler and R. S. Johannes, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," Proc. Annual Symposium on Computer Application in Medical Care, 1988.',
    "The authors used ADAP, an early neural-network-style learning method, to predict whether a woman of Pima heritage would develop diabetes within five years. The data has 768 women aged 21 or older. The paper is the original source of the Pima Indians Diabetes dataset.",
    "This is our second dataset. In it, a value of 0 for glucose, blood pressure, skin fold, insulin or BMI means the value was not measured."],
  [7, 'W. A. Wan Abu Bakar, N. L. N. Josdi, M. Man and E. A. Kadir, "Evaluating CTGAN-generated synthetic data for heart disease prediction," International Journal of Advanced Computer Science and Applications, vol. 16, no. 12, 2025.',
    "The authors trained CTGAN on the 303 Cleveland patients of the UCI heart disease data. First they filled missing values with the most common value. They then checked how close the synthetic data was to the real data, and how well models trained on it predicted disease. Training on synthetic data and testing on real data gave 56% accuracy on the five-level outcome. The study used one run and one hospital.",
    "This is the closest recent work on the same data. We compare against it in Table 2.2. We use all four hospitals, keep the gaps instead of filling them, test on unseen patients and repeat each run five times."],
  [8, 'Y. Ouyang, L. Xie, C. Li and G. Cheng, "MissDiff: Training diffusion models on tabular data with missing values," ICML 2023 Workshop on Structured Probabilistic Inference and Generative Modeling, arXiv:2307.00467, 2023.',
    "MissDiff trains a diffusion model directly on incomplete tables. The training loss simply ignores the entries that are missing. The authors also show that the usual approach, filling the gaps first and then training a generator, gives a biased picture of the real data.",
    "We cite this paper for the bias of filling gaps before generating. Our ablation study finds the same thing: with the gaps filled, our synthetic patients became much easier to spot."],
  [9, 'X. Wang, H. Asif, S. Gupta and J. Vaidya, "Data synthesis reinvented: Preserving missing patterns for enhanced analysis," IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 7, 2025.',
    "This paper argues that synthetic data should keep the missing-value patterns of the real data, because those patterns carry information. It proposes a way to keep both the observed values and the missing patterns that can be added to existing data generators.",
    "This is one of the two methods closest to ours. We share the goal. We add hospital-conditioned gaps, scores for how realistic the gaps are, and a test of their privacy cost."],
  [10, 'Z. Fang, Z. Jiang, H. Chen, X. Li and J. Li, "Understanding and mitigating memorization in diffusion models for tabular data," Proc. 42nd International Conference on Machine Learning, PMLR vol. 267, 2025.',
    "The authors measured how often diffusion models for tables copy their training records. Memorisation was common, and it grew with longer training. To reduce it, they proposed TabCutMix, a data augmentation that mixes parts of training records.",
    "This matches two of our findings. TabDDPM memorised its training patients on the diabetes data, and MedTwin started to memorise when we trained it for 1200 epochs instead of 600."],
  [11, 'E. Jang, S. Gu and B. Poole, "Categorical reparameterization with Gumbel-softmax," Proc. International Conference on Learning Representations (ICLR), 2017.',
    "Neural networks cannot easily be trained through a hard choice between categories, because the choice has no gradient. The Gumbel-softmax gives a smooth stand-in for that choice, controlled by a temperature. A low temperature makes it close to a real one-hot choice, and a 'hard' version gives an exact one-hot output while still passing gradients.",
    "MedTwin produces every category column (for example chest pain type) through a Gumbel-softmax with temperature 0.2."],
  [12, 'J. Hayes, L. Melis, G. Danezis and E. De Cristofaro, "LOGAN: Membership inference attacks against generative models," Proceedings on Privacy Enhancing Technologies, vol. 2019, no. 1, 2019.',
    "A membership inference attack tries to tell whether a given person's record was used to train a model. This paper shows such attacks on GANs. One version uses the GAN's own discriminator. Another version trains a new GAN on the released synthetic data. The attacks work best when the generator has overfitted to its training data.",
    "Our privacy tests include a membership inference attack. Our new rare-pattern attack is a version of it."],
  [13, 'T. Stadler, B. Oprisanu and C. Troncoso, "Synthetic data - anonymisation groundhog day," Proc. 31st USENIX Security Symposium, 2022.',
    "The authors attacked several synthetic data generators with linkage and attribute inference attacks. Synthetic data often protected far less than people assumed. How much privacy it gave was hard to predict in advance, and outlier records were the easiest to expose. Strong protection, such as differential privacy, cost a lot of usefulness.",
    "We cite this paper whenever we say that passing our attacks is evidence of privacy but not a guarantee. Its point about outliers fits our finding that patients with rare missing-value patterns are easier to single out."],
  [14, 'L. Xu, M. Skoularidou, A. Cuesta-Infante and K. Veeramachaneni, "Modeling tabular data using conditional GAN," Advances in Neural Information Processing Systems 32, 2019.',
    "This paper introduced CTGAN, a GAN for tables that mix numbers and categories. It normalises each number column with a mixture model, so that columns with several peaks are learned well. It also uses a conditional generator with training-by-sampling, so that rare categories are not ignored. The same paper proposed TVAE, a variational autoencoder for tables, and built a benchmark to compare generators.",
    "CTGAN and TVAE are two of our three comparison methods."],
  [15, 'A. Kotelnikov, D. Baranchuk, I. Rubachev and A. Babenko, "TabDDPM: Modelling tabular data with diffusion models," Proc. 40th International Conference on Machine Learning, PMLR vol. 202, 2023.',
    "TabDDPM is a diffusion model for tables. It uses Gaussian diffusion for number columns and multinomial diffusion for category columns, with a simple MLP network. On many benchmark datasets, it produced better synthetic data than GAN and VAE methods.",
    "TabDDPM is our third comparison method. We wrote our own compact version, following the design in this paper."],
];

const run = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 24, bold: o.bold, italics: o.italics });
const para = (children, o = {}) => new Paragraph({ alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: 276 }, keepNext: o.keepNext, children });

const body = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [run("Reference Summaries", { size: 32, bold: true })] }),
  para([run("MedTwin project report: the 15 references, in the order they are numbered in the report", { italics: true })],
    { align: AlignmentType.CENTER, after: 360 }),
];
for (const [n, cite, did, use] of REFS) {
  body.push(para([run(`[${n}] `, { bold: true }), run(cite, { bold: true })], { before: 200, keepNext: true }));
  body.push(para([run(did)], { keepNext: true }));
  body.push(para([run("In our project: ", { italics: true }), run(use)]));
}

const doc = new Document({
  creator: "Pranav Ladha", title: "MedTwin reference summaries",
  styles: { default: { document: { run: { font: FONT, size: 24 } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 22 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log(`wrote ${OUT}: ${REFS.length} summaries`); });
