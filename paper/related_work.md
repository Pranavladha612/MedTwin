# Related Work (draft)

Draft section for the MedTwin paper. Citation keys in brackets match the
reference list at the end. Every reference was checked against its
publisher or proceedings page (October 2026) except where marked
**[verify]**.

## Generative models for tabular and health data

Early GANs for patient records, such as medGAN [Choi2017], targeted
high-dimensional binary and count features from EHR codes. Mixed-type
tables, with continuous columns next to categorical ones, need more care:
CTGAN [Xu2019] introduced mode-specific normalisation for multi-modal
continuous columns and a conditional generator with training-by-sampling
for imbalanced categories, and the same paper proposed TVAE, a variational
autoencoder counterpart that it found competitive with the GAN. CTAB-GAN
[Zhao2021] extended this line to mixed and long-tailed variables.

Diffusion models have since become the strongest family on general tabular
benchmarks. TabDDPM [Kotelnikov2023] combines Gaussian diffusion for
numeric columns with multinomial diffusion for categorical ones, and TabSyn
[Zhang2024] runs score-based diffusion in the latent space of a
column-aware VAE, reporting large reductions in column-wise and pairwise
errors over earlier models. In health data specifically, EHR-Safe
[Yoon2023] uses an encoder–decoder plus GAN to handle the heterogeneity,
sparsity and variable-length sequences of real EHRs.

MedTwin builds on the Wasserstein GAN with gradient penalty [Gulrajani2017],
with Gumbel-softmax outputs for categorical columns [Jang2017] (as in
CTGAN). We benchmark against CTGAN, TVAE and TabDDPM under an identical
preprocessing, filtering and evaluation pipeline. TabSyn is not included
and is a natural next baseline.

## Missing values: imputation versus generation

Missing data is usually treated as a problem to remove before synthesis.
GAIN [Yoon2018] trains a generator to impute missing entries against a
discriminator that, given a hint vector, guesses which entries were
observed. MisGAN [Li2019] learns a mask generator alongside a data
generator, so the missingness pattern itself is modelled, and evaluates it
under missing-completely-at-random assumptions.

Our setting differs in what the output should contain. In the multi-site
heart disease data, whether a test was performed depends strongly on the
hospital (for example, cholesterol is recorded as 0 for every Swiss
patient, and thalassemia is missing for most Hungarian and VA patients).
For a synthetic cohort to be faithful, "not measured" must be part of the
generated record rather than filled in. MedTwin therefore generates a
measured/not-measured flag per incompletely observed column, conditioned
on the site, and leaves the value blank when the flag is set. This is
closer in spirit to MisGAN's mask modelling than to GAIN-style imputation,
but the mask is produced jointly with the record and its site, and is
evaluated directly (Fig. *missingness by site*). Standard tabular
generators, including all three baselines, do not accept missing values,
so we give them the same flag-plus-placeholder encoding for a fair
comparison.

## Evaluating synthetic data: fidelity, utility and privacy

**Utility.** Train-on-synthetic, test-on-real (TSTR) was introduced for
medical time series by Esteban et al. [Esteban2017] and is now the
standard utility test. We report TSTR against train-on-real (TRTR) on
real patients held out from generator training, since testing on patients
the generator saw overstates utility.

**Fidelity and detectability.** Beyond marginal and correlation
comparisons, a common fidelity check trains a classifier to tell real from
synthetic records; accuracy near 50% means the two are hard to tell apart.
We call this *detectability*. It is sometimes reported as a privacy
measure, but it is not one: a generator that copied its training records
would be undetectable. Alaa et al. [Alaa2022] make the same separation
explicit, splitting evaluation into fidelity (α-precision), diversity
(β-recall) and generalisation (authenticity, i.e. not copying training
samples).

**Privacy.** Distance to closest record (DCR), used in CTAB-GAN
[Zhao2021] among others, checks whether synthetic records sit unusually
close to real ones. We compare each record's distance to training patients
with its distance to equally many held-out patients, so that a value near
50% indicates no memorisation. Membership inference attacks on generative
models, first shown by LOGAN [Hayes2019] and systematised by GAN-Leaks
[Chen2020], ask whether an attacker can tell which patients were in the
training set; we report a distance-based attack's AUC. These are empirical
checks, not guarantees. Stadler et al. [Stadler2022] show that synthetic
data does not, in general, give a better privacy–utility trade-off than
traditional anonymisation and that its privacy gain is highly variable.
We therefore make no formal privacy claim (MedTwin uses no differential
privacy) and report the privacy metrics only as evidence of no detectable
memorisation on these datasets.

**Benchmarks.** Synthcity [Qian2023] provides a common library of
generators and metrics across tabular use cases, motivated by how
fragmented evaluation in the field has been. We follow its emphasis on
multi-metric, multi-seed evaluation; all results are means over five
seeds, each with a different held-out split, with paired confidence
intervals against each baseline.

## Datasets

The heart disease data combines four sites (Cleveland, Hungary,
Switzerland and VA Long Beach) collected for Detrano et al.
[Detrano1989] **[verify]**. The Pima Indians diabetes data comes from
Smith et al. [Smith1988].

## References

- **[Alaa2022]** A. Alaa, B. van Breugel, E. Saveliev, M. van der Schaar. How faithful is your synthetic data? Sample-level metrics for evaluating and auditing generative models. *ICML*, PMLR 162:290–306, 2022.
- **[Chen2020]** D. Chen, N. Yu, Y. Zhang, M. Fritz. GAN-Leaks: A taxonomy of membership inference attacks against generative models. *ACM CCS*, 2020. doi:10.1145/3372297.3417238.
- **[Choi2017]** E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, J. Sun. Generating multi-label discrete patient records using generative adversarial networks. *MLHC*, PMLR 68:286–305, 2017.
- **[Detrano1989]** R. Detrano et al. International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology* 64(5):304–310, 1989. **[verify — not confirmed online; check PubMed]**
- **[Esteban2017]** C. Esteban, S. L. Hyland, G. Rätsch. Real-valued (medical) time series generation with recurrent conditional GANs. arXiv:1706.02633, 2017.
- **[Gulrajani2017]** I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, A. Courville. Improved training of Wasserstein GANs. *NeurIPS*, 2017.
- **[Hayes2019]** J. Hayes, L. Melis, G. Danezis, E. De Cristofaro. LOGAN: Membership inference attacks against generative models. *Proceedings on Privacy Enhancing Technologies* 2019(1):133–152. doi:10.2478/popets-2019-0008.
- **[Jang2017]** E. Jang, S. Gu, B. Poole. Categorical reparameterization with Gumbel-softmax. *ICLR*, 2017.
- **[Kotelnikov2023]** A. Kotelnikov, D. Baranchuk, I. Rubachev, A. Babenko. TabDDPM: Modelling tabular data with diffusion models. *ICML*, PMLR 202:17564–17579, 2023.
- **[Li2019]** S. C.-X. Li, B. Jiang, B. M. Marlin. MisGAN: Learning from incomplete data with generative adversarial networks. *ICLR*, 2019.
- **[Qian2023]** Z. Qian, R. Davis, M. van der Schaar. Synthcity: A benchmark framework for diverse use cases of tabular synthetic data. *NeurIPS Datasets and Benchmarks Track*, 2023.
- **[Smith1988]** J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, R. S. Johannes. Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proc. Annual Symposium on Computer Application in Medical Care*, 261–265, 1988.
- **[Stadler2022]** T. Stadler, B. Oprisanu, C. Troncoso. Synthetic data – anonymisation groundhog day. *USENIX Security*, 1451–1468, 2022.
- **[Xu2019]** L. Xu, M. Skoularidou, A. Cuesta-Infante, K. Veeramachaneni. Modeling tabular data using conditional GAN. *NeurIPS*, 2019.
- **[Yoon2018]** J. Yoon, J. Jordon, M. van der Schaar. GAIN: Missing data imputation using generative adversarial nets. *ICML*, PMLR 80:5689–5698, 2018.
- **[Yoon2023]** J. Yoon et al. EHR-Safe: Generating high-fidelity and privacy-preserving synthetic electronic health records. *npj Digital Medicine* 6:141, 2023. doi:10.1038/s41746-023-00888-7.
- **[Zhang2024]** H. Zhang et al. Mixed-type tabular data synthesis with score-based diffusion in latent space (TabSyn). *ICLR*, 2024.
- **[Zhao2021]** Z. Zhao, A. Kunar, R. Birke, L. Y. Chen. CTAB-GAN: Effective table data synthesizing. *ACML*, PMLR 157:97–112, 2021.
