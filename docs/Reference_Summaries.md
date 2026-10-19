# Reference summaries: MedTwin project report (MUJ format)

These are the 15 references in the report, numbered as they appear in its reference list (IEEE style, in order of first citation). Each summary says what the work did, what it found, and why our project cites it.

## [1] Goncalves et al., 2020: Generation and evaluation of synthetic patient data

The authors trained several kinds of generative model on cancer registry records, from simple probabilistic models to a GAN built for medical data (MC-MedGAN). They then judged each model's synthetic data on two things: how useful it was compared with the real data, and how much it gave away about the real patients. The GAN was not the best model on every measure, and some simpler probabilistic models did as well or better. We cite this paper for the idea that synthetic patient data has to be judged on realism, usefulness and privacy together, which is how our evaluation is organised.

## [2] Goodfellow et al., 2014: Generative adversarial nets

This is the paper that introduced GANs. A generator network turns random noise into fake samples, and a discriminator network tries to tell them apart from real samples. The two are trained against each other, and the paper shows that in the ideal case the generator ends up producing samples from the real data distribution. MedTwin is a GAN, so this is the starting point for our method.

## [3] Arjovsky et al., 2017: Wasserstein generative adversarial networks

Ordinary GANs are hard to train and often collapse to producing only a few kinds of sample. This paper replaces the discriminator with a "critic" that estimates the Wasserstein (earth mover's) distance between real and fake data. Training became more stable, and the critic's loss tracked sample quality, which the original GAN loss did not. The paper kept the critic smooth by clipping its weights. We use the Wasserstein loss in MedTwin.

## [4] Gulrajani et al., 2017: Improved training of Wasserstein GANs

Weight clipping in [3] could still make training unstable. This paper replaces it with a gradient penalty that pushes the size of the critic's gradient towards 1 at points between real and fake samples. With a penalty weight of 10, training was stable across many network designs. MedTwin uses exactly this setup (WGAN-GP with a penalty weight of 10).

## [5] Janosi et al., 1989: Heart Disease (UCI Machine Learning Repository)

This is the heart disease dataset we use. It holds 920 patients from four hospitals: the Cleveland Clinic, the Hungarian Institute of Cardiology in Budapest, University Hospitals in Zurich and Basel (Switzerland), and the VA Medical Center in Long Beach. Most studies use 14 of its attributes and the presence of coronary artery disease as the target. Because each hospital recorded different tests, the missing values depend strongly on the hospital. That is the main problem our project tackles.

## [6] Smith et al., 1988: Using the ADAP learning algorithm to forecast the onset of diabetes mellitus

This paper is the source of the Pima Indians Diabetes data. The authors used ADAP, an early adaptive learning method similar to a neural network, to predict whether women of Pima heritage aged 21 or over would develop diabetes within five years, using 768 cases. In this data a value of 0 for glucose, blood pressure, skin fold, insulin or BMI means the test was not done. It is our second dataset.

## [7] Wan Abu Bakar et al., 2025: Evaluating CTGAN-generated synthetic data for heart disease prediction

This is the closest recent study on the same heart data. The authors trained CTGAN on the 303 Cleveland patients after filling missing values with the most common value, and tested fidelity, usefulness and whether important features were kept. A model trained on their synthetic data and tested on real data reached 56% accuracy on the five-level outcome. They used a single run, one hospital and no privacy test. Our study uses all four hospitals, five repeats, unseen test patients and privacy attacks.

## [8] Ouyang et al., 2023: MissDiff: Training diffusion models on tabular data with missing values

MissDiff trains a diffusion model directly on incomplete tables by ignoring the missing entries when it computes the training loss. The authors show that the common approach of filling the gaps first and then training a generator ("impute-then-generate") gives a biased picture of the data. We cite it as evidence for our own result that filling gaps makes synthetic patients easy to spot, and as one of the generators that keep missing values.

## [9] Wang et al., 2025: Data synthesis reinvented: Preserving missing patterns for enhanced analysis

This paper also keeps missing values in synthetic data. Its framework models the observed values and the pattern of missing values together, and it can be added to existing generators instead of replacing them. It shares our goal. What we add on top is hospital-conditioned gaps, scores that measure how realistic the gaps are, and a test of their privacy cost.

## [10] Fang et al., 2025: Understanding and mitigating memorization in diffusion models for tabular data

The authors show that diffusion models for tables, including TabDDPM, can copy their training records, and that this gets worse the longer the model trains. They propose a data-augmentation fix called TabCutMix. Our results match their finding twice: TabDDPM memorised its training patients on the diabetes data, and MedTwin started to memorise when we trained it for 1200 epochs instead of 600.

## [11] Jang et al., 2017: Categorical reparameterization with Gumbel-softmax

A neural network cannot learn through a hard choice of one category, because the choice has no gradient. Gumbel-softmax replaces the hard choice with a smooth version controlled by a temperature, and a "straight-through" variant gives an exact one-hot output while still letting gradients flow. MedTwin uses it for every categorical column (temperature 0.2), so each synthetic patient gets exactly one chest-pain type, one ECG result, and so on.

## [12] Hayes et al., 2019: LOGAN: Membership inference attacks against generative models

This paper showed that an attacker can often tell whether a given person's record was used to train a GAN. The white-box attack uses the trained discriminator's confidence. The black-box attack trains a separate GAN on the released synthetic data and uses that instead. The attacks worked best when the model had overfitted. Our membership inference test follows the same idea, using the distance from each real patient to the nearest synthetic record.

## [13] Stadler et al., 2022: Synthetic data – anonymisation groundhog day

The authors attacked several synthetic-data generators and found that the privacy they give is hard to predict. Either the synthetic data still leaked information about some people, especially unusual ones (outliers), or the privacy protection cost a lot of usefulness. Passing a few attacks therefore does not prove that data is safe. We cite this whenever we talk about the limits of our privacy tests. Their point about outliers also fits our finding that patients with rare missing-value patterns are slightly easier to identify.

## [14] Xu et al., 2019: Modeling tabular data using conditional GAN

This paper introduced CTGAN and TVAE, two widely used generators for tables. CTGAN handles numeric columns with several peaks through "mode-specific normalisation", and handles rare categories by conditioning the generator on a chosen category during training. TVAE is a variational autoencoder adapted to tables. Both are baselines in our comparison, run through the open-source ctgan library.

## [15] Kotelnikov et al., 2023: TabDDPM: Modelling tabular data with diffusion models

TabDDPM applies diffusion models to tables. It adds Gaussian noise to numeric columns and multinomial noise to categorical ones, then trains a network to remove that noise step by step. On many benchmark datasets it beat GAN and VAE generators on how useful its synthetic data was. TabDDPM is our third baseline; we wrote our own compact version following the paper's design.
