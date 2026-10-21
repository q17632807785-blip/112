# Preregistration: A Specificity-Matched Benchmark of Denoising Methods on Public Octopus G1 Visual Fields (GRAPE, Direction A)

> **English translation.** This is a full translation of `preregistration.md` (Simplified Chinese). In case of any discrepancy, the Chinese version is authoritative. Numbers, mathematical expressions and tables are reproduced verbatim; the YAML block in Part 5 is reproduced verbatim (including its Chinese comments) and is byte-for-byte identical in content to `config.yaml`.

> **Status: v2 published on 2026-10-20 (GitHub tag `v2.0-prereg`, Zenodo DOI 10.5281/zenodo.23494626); later revisions R5, R6, R7, R8, R9 and R10 are listed in section 7.** To be published by the user in the public GitHub repository https://github.com/q17632807785-blip/112: after this file is committed, a tag and a release are created, and the release is archived on Zenodo to obtain a DOI (GitHub commit times can be rewritten; the Zenodo archive provides an immutable timestamp). The repository does not contain the raw GRAPE data or per-person derived data (`data/*.parquet`, `split.json`); it contains only this file, `config.yaml`, `g1_locations.csv` and the code; Part 6 gives the SHA-256 of the files that are not made public.
> Date generated: 2026-10-20 (Asia/Shanghai). Basis: `方向A_研究计划_最终版.md` (Direction A research plan, final version; hereafter "the plan") and the 2026-10-20 revisions R1 and R2.
> This file supersedes `preregistration_draft.md`. Before publication, `config.yaml` must be synchronised verbatim with Part 5.

## 0. Title, data, positioning

- Data: GRAPE (Huang et al., Sci Data 2023;10:520), 263 eyes / 144 subjects / 1115 Octopus 900 G1 visual field examinations (Second Affiliated Hospital of Zhejiang University).
- Positioning:

> On the only public longitudinal Octopus G1 dataset, we establish a specificity-matched benchmark of visual field denoising methods whose primary endpoint is slope error against simulated ground truth, comparing statistical models with a masked denoising model that handles 0 dB censoring, examination-level global offsets and spatiotemporal correlation.

- Additional related work: Gwak & Kang, "When to Denoise Longitudinal Visual Field Labels: Anchor-Preserving Supervision under Short Follow-up", MICCAI 2026 OMIA Workshop (open-access full text: https://papers.miccai.org/miccai-2026-sat/paper/OMIA_015.pdf). On the 263 GRAPE eyes, that paper applies **empirical Bayes shrinkage to the pointwise slopes of the follow-up segment (shrinking towards the pointwise population mean of the training eyes)**, solely to provide training labels for a ResNet-50 mapping "a single fundus photograph → the first visual field"; with eye-level splitting and 3 examinations the MAE improves slightly by 0.075 dB, and the gain disappears with subject-level splitting. It does not address slope accuracy, simulated ground truth, specificity-matched detection, or censoring modelling. The EB comparator in this study (shrinking towards the mean of the same cluster in the same eye) has a different shrinkage target from that paper; EB-GK is additionally included as a descriptive comparator (Section 2.2).

### 0.1 Knowledge of the data before publication (disclosure)

The investigators had access to the GRAPE data before publishing this preregistration, as detailed below. Apart from this, no competing method has been run on simulated data, and no comparison between methods has been produced.

1. **v2 project** (2026-10-20 00:39–03:41): on the same data, a linear mixed model was used to test whether the fundus photograph at the first visit explains the rate of visual field decline (244-eye definition); the primary test gave a subject-level permutation p = 0.085. That project computed the MS residual SD, slope reliability and the signal ceiling.
2. **Data facts** (Section 1 of the plan): on **all 263 eyes**, summary statistics were computed, including the distribution of the number of examinations, the censoring proportion, MS residual SD, slope SD and standard error, pointwise residual SD, examination-level global offset SD, and inter-point residual correlations. These computations took place before the subject split, so the held-out subjects entered these numbers in summary form; no result for any individual held-out eye was viewed.
3. **Stage 1 data steps** (2026-10-20 16:15–16:35): construction of the long table, G1 location mapping and its three validations (the neighbouring-point residual correlation was computed using all examinations), the subject split, and the noise model (development set only). The noise model script includes a self-check simulation: data are generated from the noise model and the global offset SD and the censoring proportion are re-estimated (results in Part 6, item 4); this concerns only the noise model itself and no competing method was run.
4. **Pipeline smoke test** (2026-10-20 11:50–12:35): before publication, a single pipeline smoke test was run on the GRAPE data, solely to check whether the code could run end to end from scratch. The outputs of that test were not sealed and were deleted on 2026-10-20; the investigators never viewed any of the values in them. The code of that test is not the implementation described in this preregistration, and its outputs are not used in any analysis.

---

## 1. Research questions, endpoints, testing order

### 1.1 Primary endpoint H1

**H1**: On `sim_test`, with the slope of every simulated series estimated using all 4 examinations, the **root mean squared error of the cluster-level slopes** (10 clusters, dB/year) is lower for G1-MST than for B4.

- **Computation of the RMSE**: for each scenario \(s\), \( \mathrm{MSE}_s = \) the mean of \( (\hat b - b_{true})^2 \) over all series of that scenario × 10 clusters (the 10 clusters are equally weighted, not weighted by number of points); the 8 scenarios are pooled with equal weights: \( \mathrm{RMSE} = \sqrt{\tfrac18 \sum_s \mathrm{MSE}_s} \). \( \sqrt{\mathrm{MSE}_s} \) is also reported per scenario.
- **Seeds**: for G1-MST, the RMSE is computed separately for each of the 5 seeds and the 5 RMSEs are then averaged (the estimates of the 5 seeds are not first averaged into an ensemble). Methods that do not require training seeds are computed once.
- **Statistic**: the difference \( \Delta = \mathrm{RMSE}_{G1\text{-}MST} - \mathrm{RMSE}_{B4} \).
- **Interval**: the held-out subjects (43 subjects) are resampled with replacement 2000 times; each resample includes all series generated from all template eyes of the drawn subjects, and \( \Delta \) is recomputed; the 2.5% and 97.5% quantiles form the 95% interval.
- **Decision**: if the upper limit of the interval < 0, G1-MST is better (H1 holds); if the interval contains 0, the two cannot be distinguished; if the lower limit > 0, B4 is better. In the latter two cases H1 does not hold.

### 1.2 Key secondary endpoints and fixed order

The testing order is fixed as **H1 → H3 → H4 → H2**. "Holds" for each item means that G1-MST is superior to B4 (the 95% interval lies entirely on the side favouring G1-MST). The next item is formally tested only if the preceding one holds; once an item does not hold, the subsequent items are still computed and reported, but only descriptively, without concluding "better". Each item uses a two-sided 95% interval (equivalent to one-sided α = 0.025), and the fixed sequence guarantees that the overall type I error rate does not exceed 0.025 (one-sided).

- **H3**: on `sim_test`, at k = 4, the detection sensitivity at 95% specificity in the focal progression scenarios (S2, S3, S4, S6, S7 pooled), G1-MST versus B4. The decision statistic, threshold and interval algorithm are given in Section 3.6.
- **H4 (real-data anchor)**: for real eyes with at least 5 examinations (88 eyes), the cluster-level trend estimated from the first 4 examinations is extrapolated to predict the observed cluster means at the 5th and subsequent examinations; the MAE is compared (G1-MST − B4). The algorithm is given in Section 3.7.
- **H2**: root mean squared error of the global slope (average of 59 points), G1-MST versus B4, with the same algorithm and decision rule as H1 (replacing "10 clusters" by a single global value).

**Rationale for placing H2 last (expected to be indistinguishable)**: the noise in the global slope comes mainly from the whole examination shifting together — the global offset SD per examination is about 1.45 dB, whereas it would be only 0.44 dB if the noise at the 59 points were independent; borrowing information from neighbouring points within the same examination cannot remove a common offset, and the shrinkage of a linear mixed model is already close to optimal. Moreover, after removing the global offset, the residual correlation between neighbouring points in the same hemifield at distance ≤ 6° reaches 0.507 (Part 6, item 3), indicating that the pointwise noise itself is spatially correlated (possibly related to the Octopus TOP strategy estimating thresholds using responses at neighbouring points; this is only a conjecture), which further reduces the room for spatial borrowing. The figure "mean inter-point correlation after removing the offset −0.017" quoted in Section 0 of the plan is an algebraic consequence of row-centring (about −1/58) and is not used as evidence for anything.

### 1.3 Scale of the ground truth

- **Primary analysis (observed scale)**: the true trajectory of each point is first truncated below 0 dB to 0, i.e. \( \max(f(t), 0) \), and OLS is then performed on the examination times within the observation window to obtain the true pointwise slope; the true cluster-level and global values are the averages of the true values of the corresponding points. H1–H4 and the slope endpoints of Section 1.4 all use this scale.
- **Descriptive analysis (latent scale)**: the OLS slope of the untruncated true trajectory \( f(t) \). Reported only in supplementary tables.
- Rationale: after revision R1 the template baseline may lie below 0 dB; a point that remains below 0 dB throughout is always observed as 0, and its latent slope cannot be recovered by any method; with the observed scale as the primary endpoint, what is compared is each method's ability to recover slopes from observable information.

### 1.4 Other secondary and exploratory endpoints (descriptive only, no formal testing)

- For each method, each scenario and each number of examinations (k = 3, 4, 5, 6): bias, root mean squared error and 95% interval coverage (raw and calibrated standard errors) of pointwise, cluster-level and global slopes. Stratified by true rate, with bias reported specifically for fast-progressing eyes.
- The same set of metrics under the latent-scale ground truth (Section 1.3).
- Real data: detection proportions and Kaplan–Meier curves after calibration to a 5% false-positive rate using shuffled visits (Section 3.8).
- Real data: pointwise MAE of next-visual-field prediction (history lengths 2 and 3), stratified by tertiles of baseline MS and of the full-series OLS slope, always including "last value carried forward" (B0).
- For the 50 eyes with at least 6 examinations: odd–even split-half slope reliability (intraclass correlation coefficient).
- Structure–function correlation between the denoised baseline visual field and the four RNFL sectors (Garway-Heath mapping, Spearman correlation, raw versus denoised).
- Agreement with the official PLR2, PLR3 and MD labels (supplementary table).
- Results of EB-GK on all of the above endpoints (Section 2.2).

### 1.5 Sample size statement

This study does not perform a formal power calculation. The size of `sim_test` (78 eyes × 8 scenarios × 20 repetitions) is fixed before generation; the effective resampling unit for the primary endpoint interval is the 43 held-out subjects, the precision of the conclusions is limited by the template diversity of these 43 subjects, and this will be reported truthfully in the paper. The real-data endpoint H4 has only 88 eyes, and its interval is expected to be wide.

---

## 2. Methods and comparators

### 2.1 Competing methods

**First batch (must be completed):**

| ID | Method | Key points |
|---|---|---|
| B0 | Last value carried forward | Used only for prediction endpoints |
| B1 | Pointwise linear regression (PLR) | Used only with ≥3 examinations |
| B2 | MS linear regression | Global |
| B3 | Linear mixed model (MS) | Two-level (subject, eye) random intercepts and slopes |
| B4 | Spatial two-level mixed model | Following Zhao 2026: eye-level plus cluster-level random intercepts and slopes, plus an examination-level random intercept; **with a spatial term** (Part 6, item 3) |
| EB | Empirical Bayes shrinkage of pointwise slopes | Pointwise OLS slopes shrunk towards the cluster mean, weight τ²/(τ²+SE²), τ² estimated from training eyes, stratified by sensitivity |
| B5 | Spatial smoothing followed by PLR | Nearest-neighbour smoothing, simple to implement |
| MST-S | G1-MST without temporal attention | Equivalent to the single-examination masked autoencoder of Wu 2024 |
| G1-MST | Proposed model | Section 2.5 |
| D1 | PoPLR | Used only for detection endpoints |

**Second batch (if resources allow)**: B6 VAE denoising, B8 Kalman filter, B9 small GRU, B10 deterministic pointwise-token Transformer (point-estimate version of Bern 2026). Any not done must be stated in the paper with reasons. None of the second-batch methods enters H1–H4.

### 2.2 Descriptive comparator EB-GK (revision R2)

| ID | Method | Key points |
|---|---|---|
| EB-GK | Gwak–Kang-style empirical Bayes shrinkage of pointwise slopes (descriptive comparator) | For each point of each eye, use the full-series OLS slope \( b \) and standard error \( s \); shrink towards **the population mean slope of the training eyes at that location** \( \mu_p \); the prior variance is estimated per location \( \tau_p^2 = \max(0,\ \mathrm{var}_{train}(b_p) - \overline{s_p^2}) \); \( b_{GK} = \mu_p + w(b - \mu_p) \), \( w = \tau_p^2/(\tau_p^2 + s^2) \); parameters are estimated on training eyes only. Cluster-level and global slopes are obtained by averaging the shrunk pointwise slopes as in Section 2.3 |

- Source: Gwak & Kang, MICCAI 2026 OMIA (OMIA_015). The original paper shrinks only the follow-up-segment slopes and keeps the first visual field as an "anchor", in order to provide labels for a fundus-photograph model; this study uses full-series slopes so that it can be compared with the other methods on the same slope endpoints.
- Difference from EB: EB shrinks towards the mean of the same cluster in the same eye, with τ² estimated in three sensitivity strata; EB-GK shrinks towards the training-population mean at that location, with τ² estimated per location.
- **EB-GK is a descriptive comparator only**: it does not enter the fixed-sequence tests H1–H4 and does not take part in the decision of Section 4.2.

### 2.3 Unified interface and slope definitions

Each method is implemented as a class that receives a (truncated) series and outputs:

```python
class Method:
    def fit(self, train_series: list[Series]) -> None: ...
    def estimate(self, series: Series) -> Estimate: ...
    # Estimate 字段：
    #   denoised: (n_visits, 59) 去噪值（没有去噪的方法就是原值）
    #   slope_point, se_point: (59,)
    #   slope_cluster, se_cluster: (10,)
    #   slope_global, se_global: float
    #   forecast(t_new) -> (59,) 预测值，可选给 sigma
```

(Code comments translated: `Estimate` fields — `denoised`: (n_visits, 59) denoised values (for methods that do not denoise, the raw values); `slope_point, se_point`: (59,); `slope_cluster, se_cluster`: (10,); `slope_global, se_global`: float; `forecast(t_new) -> (59,)` predicted values, optionally with sigma.)

`Series` fields: `eye_id, subject, t (n,), y (n, 59), censored (n, 59) bool, age (n,), iop (n,)`. Censored observations are recorded as 0 dB with `censored = True`.

- **Cluster-level slope** (identical for all methods): first average the denoised values of each examination within each cluster, then perform OLS against time. The central point (0,0) does not belong to any cluster.
- **Global slope**: average the 59 points, then perform OLS against time.
- Methods such as B4, EB and EB-GK, which directly produce cluster-level or pointwise slopes, use their own estimates; this is stated in the supplementary tables.
- Whenever any method estimates slopes or makes a decision, only the series truncated at the current examination is passed to `estimate()`.
- **Estimation failure**: when a method fails to estimate a given series (non-convergence, numerical overflow, etc.), the B1 estimate for that series is used in that method's results instead, and the failure rate is reported by method × scenario.

### 2.4 Implementation of comparators

| Method | Key implementation details |
|---|---|
| B0, B1, B2 | B1: pointwise OLS, standard error from that point's residuals; B2: OLS on MS |
| B3 | statsmodels `mixedlm`: subject as group, within-subject random intercept and slope, eyes nested. For a new eye, the variance components are fixed at the training-set values and only the best linear unbiased prediction (BLUP) for that eye is computed |
| B4 | Fixed effects: intercept + time. Random effects: eye (intercept, slope), cluster within eye (intercept, slope), examination (intercept), plus a spatial correlation term for the pointwise residuals. Workflow: estimate variance components on training eyes → for a new eye, fix the variance components and compute BLUPs, giving cluster-level slopes and their standard errors |
| EB | Pointwise OLS slope \( b \) and standard error \( s \); in three sensitivity strata, the prior variance is estimated from training eyes \( \tau^2 = \max(0,\ \mathrm{var}(b) - \overline{s^2}) \); shrink towards the mean slope of the same cluster in the same eye: \( b_{EB} = w b + (1-w)\bar b_{cluster} \), \( w = \tau^2/(\tau^2 + s^2) \) |
| B5 | Within each examination, each point is replaced by a weighted average of itself and its neighbours in the same hemifield within ≤6° (own weight 0.5, the remaining 0.5 split equally among neighbours), followed by PLR |
| D1 PoPLR | Pointwise OLS one-sided p-values; statistic \( S = -\sum \ln p \) (only over points with negative slopes); the visit order is randomly permuted 1000 times to obtain the permutation distribution. If R is available, results for 10 eyes are checked against `visualFields` |

**Implementation of the B4 spatial term and fallback (rules fixed in advance)**:
1. Preferred: Matérn spatial correlation in R `spaMM` (only within a hemifield; no correlation across the horizontal midline).
2. In week 3 (2026-11-02 to 11-08), run time is measured on 20 randomly chosen development-set eyes and extrapolated linearly by number of eyes to the time required to estimate variance components on all development-set training eyes; if the projected time exceeds 24 hours, a conditional autoregressive structure on the adjacency graph (neighbours = same hemifield, distance ≤ 6°) is used instead.
3. If variance-component estimation does not converge, in order: ① make the random-effects covariance diagonal; ② if it still does not converge, switch to the conditional autoregressive structure.
4. The option finally adopted is appended as a timestamped revision before any method results are produced, and is not changed thereafter.

### 2.5 G1-MST

**Architecture (about 30,000 parameters, fixed):**

- One token per point value. Embedding = linear layer(`[y/35, censored]`) + Fourier features of the coordinates (x, y) (8 frequencies) through a linear layer + Fourier features of time (years) through a linear layer + age at visit (scalar) through a linear layer. Dimension d = 32.
- Each examination additionally has an "examination token" (a learnable initial vector plus time encoding), used to represent the global offset of that examination.
- 2 layers of factorised attention, each layer:
  - Spatial attention: among the 59 point tokens plus 1 examination token within the same examination. Attention bias: point pairs in the same hemifield at distance ≤ 6° receive +b (b learnable); point pairs across the horizontal midline receive −∞ (not connected). The examination token is connected to all points.
  - Temporal attention: for the same point across different examinations; examination tokens also attend to each other temporally.
  - Feed-forward width 64, LayerNorm, dropout 0.1.
- Output head: each point token outputs \( \mu \) and \( \log\sigma \).

**Training objective:**
- In each batch, one masking mode is chosen at random (equal probabilities) for each series: ① randomly mask 20% of the point tokens; ② mask an entire examination (all 59 points); ③ mask all examinations of the same point.
- Masked tokens have their value and censoring flag set to zero, and a "masked" flag is added.
- The loss is computed only on masked tokens, using the censored Gaussian negative log-likelihood: uncensored \( -\log \mathcal N(y;\mu,\sigma) \); censored \( -\log \Phi\big((0-\mu)/\sigma\big) \).
- Optimisation: AdamW, learning rate 1e-3, weight decay 1e-2, batch size 16 series, at most 400 epochs; early stopping on the masked-token negative log-likelihood on the validation folds of a 5-fold split within the development set (patience 30 epochs).
- Training data: real data only (training folds of the development set). **The main model is not trained on simulated data.** A separate "real + `sim_train`" version (G1-MST-sim) is trained, only for sensitivity analysis SA7.
- 5 seeds: 20261019–20261023.

**Inference (obtaining denoised values):**
- For a (truncated) series, K = 20 inference passes are run, each randomly masking 20% of the point tokens, with every token masked at least 2 times. The denoised value of each token = the average of its \( \mu \) over the passes in which it was masked. Thus no denoised value uses that point's own measurement at that examination (the J-invariance principle, Batson & Royer 2019).
- Forecasting: a fully masked examination is added at the target time and \( \mu, \sigma \) are read out (used only for the next-visit prediction in Section 1.4, not for H4).

**Ablations** (only MST-S in the first batch; the others after the main results, all descriptive): remove temporal attention (MST-S), remove the examination token, remove the censored likelihood (use MSE, −1 treated as 0), remove heteroscedastic σ, random-point masking only, add follow-up IOP, a 200,000-parameter version (d = 64, 3 layers).

### 2.6 Calibration of slope standard errors

Denoised values borrow information from each other, so standard errors computed directly with the OLS formula will be too narrow. Procedure:
1. On `sim_train`, for each method and each level (pointwise, cluster, global), compute \( z = (\hat b - b_{true})/\widehat{SE} \) (\( b_{true} \) on the observed scale).
2. Calibration factor \( c = \) the 95% quantile of \( |z| \) / 1.96.
3. On `sim_test` and on real data, \( c \cdot \widehat{SE} \) is used as the calibrated standard error.
4. Both raw and calibrated coverage are reported.

All methods are calibrated in this way (including B1 and B4), to ensure a fair detection comparison.

---

## 3. Data, noise model, simulator and evaluation workflow

### 3.1 Data encoding

- Indices 21 and 32 of the 61 GRAPE values (physiological blind-spot placeholders) are removed, leaving 59 points.
- All other −1 values are left-censored observations at 0 dB: `censored = True`, value recorded as 0 dB.
- Locations, clusters and the split were fixed before publication; see Part 6.

### 3.2 Subject split rules

- The noise model, simulator parameters, tuning of all methods, and `sim_train` use only development-set subjects.
- `sim_test` uses only held-out subjects as templates.
- G1-MST and the other methods that require training, when used for simulation evaluation, are trained only on development-set real data.
- Real-data endpoints (H4 and the real-data analyses of Section 1.4) use a 5-fold subject-level split of all 144 subjects (`all_folds`); in each fold every method is retrained on the other 4 folds.

### 3.3 Noise model (development set only)

For each eye and each point, OLS gives fitted values \( \hat f \) and residuals \( r \); the residuals are multiplied by \( \sqrt{n/(n-2)} \).

1. **Examination-level global offset**: for each examination \( \delta_v = \) the mean of the residuals of its 59 points; stratified by tertile of the eye's baseline MS (the three strata differ by more than 20%; see Part 6, item 4). Simulation uses the deconvolved SD, after removing the average contribution of the pointwise noise.
2. **Point-specific noise**: \( e = r - \delta_v \). The SD is estimated in bins of the fitted value (0–5, 5–10, 10–15, 15–20, 20–25, 25–30, ≥30 dB), and a smoothing spline is then fitted to SD(\( \hat f \)).
3. **Spatial correlation**: within the same hemifield, pointwise noise is generated with an exponentially decaying spatial covariance, with the correlation length estimated from the data; no correlation across the horizontal midline. When generating noise, the central point is merged into the superior-hemifield block.
4. **Censoring**: simulated values < 0 are recorded as 0 with `censored = True`.

### 3.4 Simulator

Each simulated series = true trajectory + noise.

**Templates (revision R1)**: for each template eye, the **intercept of a pointwise Tobit regression (left-censored at 0 dB)** is taken as the latent baseline at the time of the first examination (t = 0); it may be below 0 dB and is restricted to −10–35 dB.
- For each point of each template eye, fit \( y^* = a + b\,t + \varepsilon \), with observations \( \max(y^*, 0) \) and −1 treated as \( y^* \le 0 \); \( \sigma \) is not estimated per point but fixed at the noise-model value \( \sqrt{SD_{point}(\hat f_{OLS})^2 + SD_{offset}^2} \) (the offset SD taken from the three baseline-MS strata); \( a, b \) are estimated by maximum likelihood. For a point with no censoring this equals OLS; when all examinations are censored the MLE diverges and the lower bound −10 dB is used.
- Rationale: with OLS intercepts (floored at 0 dB) as templates, the simulated censoring proportion in the noise-model self-check was 2.75%, versus 4.15% in the real development set, beyond ±1 percentage point; 35% of censored points are "censored at every examination of that eye" absolute scotomas, which OLS fits to about 0 dB and of which only about half fall below 0 in simulation. A diagnostic simulation setting only such points to −10 dB raised the proportion to 3.46%, indicating that the main cause is the 0 dB floor.

**Schedule**: the primary evaluation requires "exactly 4 examinations":
- If the template eye has ≥4 examinations, its first 4 time points are used; if fewer than 4, intervals are drawn with replacement from the empirical distribution of real consecutive intervals in the development set until there are 4 (`sim_test` also draws from the development-set interval distribution).
- Versions with k = 3, 5 and 6 examinations are also generated (intervals drawn in the same way), for the "number of examinations – error" curves.

**True-slope scenarios (8):**

| ID | Name | True trajectory | Positive/negative | "Favourable to B4"? |
|---|---|---|---|---|
| S0 | Stable plus age-related decline | All points −0.1 dB/year | Negative | Yes |
| S1 | Cataract-like | All points drop by 1–2 dB at once after a random examination, stable thereafter | Negative | No |
| S2 | Focal progression within clusters | All points in 1–2 random clusters −0.5 / −1 / −2 dB/year (three levels equally likely); the rest as S0 | Positive | Yes |
| S3 | Cross-cluster arcuate defect | An arc along the nerve fibre course (spanning 2–3 clusters, but covering only part of the points in each cluster) −1 / −2 dB/year | Positive | No |
| S4 | Small paracentral scotoma | 3–5 adjacent points within the central 10° −1.5 / −3 dB/year | Positive | No |
| S5 | Diffuse progression | All points −0.3 / −0.6 dB/year | Positive | Yes |
| S6 | Late acceleration | Same locations as S2, −0.3 in the first half and −1.5 dB/year in the second half (piecewise linear, knot at the time midpoint) | Positive | No |
| S7 | Quadratic acceleration | Same locations as S3, trajectory \( -a t^2 \), with \( a \) such that the mean slope at the end of 4 examinations is −1 dB/year | Positive | No |

**Supplementary rules for the scenarios**:
- "1–2", "3–5" and "2–3" in the table are integers drawn with equal probability; discrete levels such as "−1 / −2" are equally likely; the drop size in S1 is drawn uniformly in [1, 2] dB.
- S1: the drop occurs before the \(j\)-th examination, with \(j\) drawn with equal probability from \(\{2, \dots, k\}\); the slope is 0 both before and after the drop (the S0 age-related decline is not added). By Section 1.3, the observed-scale true slope of S1 is generally negative, but in the detection endpoints it is counted as **negative** (used to set the specificity threshold); the slope-error endpoints use its true value as usual.
- In S2, S5 and S6, unaffected points are as in S0 (−0.1 dB/year).
- The point sets of the S3 arc and the S4 scotoma are generated automatically by the simulator code from the coordinates and the 10-cluster definition. After the simulator code is finalised and before any simulated data are generated, the point-set generation rules and the code SHA-256 are appended as a timestamped revision.

**Size and seeds**:
- `sim_train`: each development-set template eye × 8 scenarios × 10 repetitions, seed 20261019.
- `sim_test`: each held-out template eye × 8 scenarios × 20 repetitions, seed 20261119.
- Each series carries `scenario, template_eye, template_subject, truth_point, truth_cluster, truth_global (one set each on the observed and latent scales), is_progressing`.

**Sealing of `sim_test`**: immediately after generation, the file SHA-256 is appended to the preregistration as a timestamped revision; **before 2027-01-04 (the first day of week 12)** no script may read it (the `sim_test_seal` switch in `config.yaml`; evaluation scripts check the switch and the hash). Weeks are counted with the week containing 2026-10-19 (Monday) as week 1.

### 3.5 Simulator calibration and the week-5 checkpoint (development-set real data and `sim_train` only)

| Statistic | Allowed deviation |
|---|---|
| MS residual SD | ±10% |
| Examination global offset SD | ±10% |
| Pointwise residual SD (stratified by fitted value, each stratum) | ±15% |
| SD of observed MS slope | ±15% |
| Censoring proportion (target: 4.15% observed in the development set) | ±1 percentage point |

- **Mixing rule for the observed MS slope SD**: each development-set eye is classified by its full-series OLS MS slope into three bands: ≥ −0.3 dB/year, −1.0 to −0.3 dB/year, < −1.0 dB/year; the proportions of eyes in the three bands are used as the mixing weights of S0, S5 and S2 respectively; the SD of the observed MS slope is computed from the S0, S5 and S2 series of `sim_train` weighted accordingly and compared with the real development-set value.
- If the allowed deviation is exceeded, the noise model and simulator parameters are adjusted, for at most 3 rounds; if still out of tolerance after tuning, the gap is reported in the paper. Tuning looks only at the statistics in the table above and does not run any competing method.
- **Week-5 checkpoint** (2026-11-16 to 11-22): on `sim_train`, the cluster-level slope RMSE of B4 should be clearly lower than that of B1 (bootstrap over development-set subjects, interval excluding 0). If not achieved, the simulator or the B4 implementation is fixed first before proceeding.

### 3.6 Detection on simulated data (H3)

1. Decision statistic of each method: the minimum over clusters of the z-values of the cluster slopes (using calibrated standard errors); PoPLR uses S.
2. On the negative scenarios of `sim_train` (S0, S1), find the threshold at which 95% of negative series are not called positive, separately for each method × each seed × each k.
3. Apply the thresholds to `sim_test`: report the actual specificity in the negative scenarios and the sensitivity in the positive scenarios, stratified by scenario and rate.
4. H3: the difference in sensitivity (G1-MST − B4; G1-MST takes the mean sensitivity over its 5 seeds) at k = 4 in the focal progression scenarios (S2, S3, S4, S6, S7 pooled, each scenario equally weighted). The 95% interval is obtained by resampling held-out subjects with replacement 2000 times, with thresholds held fixed (not re-estimated) during resampling.
5. Additionally reported: "number of examinations required to reach 80% sensitivity" (curve for k from 3 to 6).

### 3.7 Real-data anchor (H4)

1. Subjects: eyes with at least 5 examinations (88 eyes); 5-fold subject-level split of all 144 subjects, each fold using models trained on the other 4 folds.
2. Each method calls `estimate()` on the first 4 examinations to obtain the intercept and slope of each cluster: general methods perform OLS on the denoised cluster means of the 4 examinations; B4, EB and EB-GK use their own cluster-level estimates. These are extrapolated to the times of the 5th and subsequent examinations. G1-MST does not use its forecast output but performs the same cluster-level trend extrapolation as the other methods; for G1-MST, the average of the extrapolated values over its 5 seeds is used as the prediction.
3. Observed cluster mean: the average of the observed values (censored recorded as 0 dB) of the points of that cluster at that examination.
4. MAE: for each eye, first average \( |\text{predicted} - \text{observed}| \) over its 5th and subsequent examinations × 10 clusters, then average over eyes with equal weights.
5. Interval: resample subjects with replacement 2000 times to obtain the 95% interval of the MAE difference (G1-MST − B4).

### 3.8 Real-data detection (descriptive only)

1. For eyes with at least 5 examinations, the visit time labels of each eye are shuffled 200 times (time points unchanged, only the order of values permuted), and each method's decision statistic is computed.
2. Threshold: the positive rate on shuffled series equals 5%.
3. On the real order, decisions are made at each examination starting from the 3rd, recording the examination number and years at first detection; Kaplan–Meier curves are drawn, with a subject-clustered log-rank test.
4. Limitation: shuffling erases the age-related decline and cannot separate declines caused by cataract from those caused by treatment.

---

## 4. Sensitivity analyses and narrative decision point

### 4.1 Sensitivity analyses (all descriptive; they do not change the decisions on H1–H4)

| ID | Change |
|---|---|
| SA1 | −1 treated as missing (v1 encoding), to assess the effect of censoring handling |
| SA2 | Exclude angle-closure eyes |
| SA3 | Generate `sim_test` only from template eyes with ≥4 examinations |
| SA4 | Exclude eyes whose IOP during follow-up fell by more than 30% from baseline (the data have no medication records; this serves as a proxy for treatment intensification) |
| SA5 | Detection specificity changed to 90% and 97.5% |
| SA6 | G1-MST 200,000-parameter version; masking proportions 10% and 30% |
| SA7 | G1-MST-sim (trained with added simulated data) |
| SA8 | Simulated noise without the examination global offset (pointwise noise only), to see whether the conclusions depend on this component |
| SA9 | If Octopus age norms are available, redo H1 with TD values |
| SA10 | Simulated pointwise noise made spatially independent, to see whether the conclusions depend on spatially correlated noise |

### 4.2 Week-11 narrative decision point (2026-12-28 to 2027-01-03)

Using only `sim_train` and the development set: the difference in cluster-level slope RMSE between G1-MST and B4, with a paired bootstrap over development-set subjects (same algorithm as H1).

- Upper limit of the interval < 0: the abstract and title emphasise "a spatiotemporal denoising model outperforms a strong statistical model on cluster-level slopes".
- Interval contains 0 or B4 is better: the title and abstract are written as a "benchmark", emphasising "with small samples a strong statistical model is sufficient; the global offset dominates global noise".

**In either case, none of the endpoints, the testing order or the decision rules of Part 1 are changed**; only the narrative emphasis of the paper is decided. The decision and its date are recorded in the log.

---

## 5. Hyperparameters fixed in advance (`config.yaml` must match this verbatim)

> Note (translation): the YAML block below is reproduced verbatim from the Chinese preregistration, including its Chinese comments, and is identical in content to `config.yaml`. It is deliberately not translated, so that it remains the single authoritative configuration.

```yaml
# 方向 A：GRAPE 视野去噪方法基准 —— 全局配置（预先定死，预注册内容的一部分）
# 依据：方向A_研究计划_最终版.md（2026-10-20）。修改任何超参数都必须在预注册里带时间戳记录。
project: GRAPE_directionA_VF_denoising_benchmark
plan_version: "最终版 2026-10-20"
created: "2026-10-20"
revisions: ["R1 2026-10-20 Tobit 模板基线", "R2 2026-10-20 EB-GK 描述性对照", "R3 2026-10-20 发布前重新生成数据步骤文件并更新第 6 部分", "R4 2026-10-20 以本版本整体替换已被取代的草稿 3129355", "R5 2026-10-20 S3/S4 点集生成规则、模拟器未定细则与模拟器代码 SHA-256（生成任何模拟数据之前）", "R6 2026-10-20 更正文首状态行：v2 已发布（DOI 10.5281/zenodo.23494626）", "R7 2026-10-21 sim_test SHA-256、最终校准 κ 与 sim_calibration.json SHA-256", "R8 2026-10-21 B4 空间项锁定为 spaMM Matérn（同半视野；第 3 周计时提前于 2026-10-21 完成）", "R9 2026-10-21 第 5 周检查点通过（B4 vs B1）；披露提前运行；更新 lmm.py SHA-256（spatial_impl 未改）", "R10 2026-10-21 B4 逐点 SE 修复；更新 lmm.py SHA-256（spatial_impl 与簇级检查点未改）"]

paths:
  raw_xlsx: "../VF and clinical information.xlsx"   # 相对 directionA/；原始数据不复制进 data/
  data: data
  results: results
  figures: figures
  logs: logs

seed: 20261019               # 全局种子；所有随机步骤从这里读并写进日志
model_seeds: [20261019, 20261020, 20261021, 20261022, 20261023]
week1_start: "2026-10-19"    # 第 1 周从这一天（周一）算起

data:
  n_vf_values_raw: 61
  blind_spot_idx: [21, 32]   # 删除这两个占位格后剩 59 点
  censor_code: -1            # 除盲点外 −1 = 0 dB 处左删失
  censor_value_db: 0.0
  expected:
    n_eyes: 263
    n_subjects: 144
    n_visits: 1115
    n_point_values: 65785
    n_censored: 2734

locations:
  file: data/g1_locations.csv
  coord_source: "R 包 visualFields 1.0.7 locmaps$g1（Octopus G1，右眼视野坐标，颞侧为 +x）"
  order_source: "GRAPE 原文图 2b（从中心向外螺旋），用 8 号右眼第 1 次检查逐点核对"
  cluster_source: "Octopus EyeSuite G 程序 10 簇（Bürki & Monhart 2007, Eur Ophthalmic Rev, 图 1b）"
  center_point_cluster: none # 中心点 (0,0) 不属于任何簇；簇级终点不含它，全局终点含它
  neighbor_max_dist_deg: 6.0 # “相邻点”= 同一半视野、距离 ≤ 6°
  neighbor_resid_corr: 0.5067
  other_resid_corr: -0.0353
  neighbor_resid_corr_threshold: 0.10

split:
  unit: subject
  dev_frac: 0.70
  holdout_frac: 0.30
  strata: ["baseline_ms_tertile", "has_eye_ge5_visits"]
  dev_n_folds: 5
  all_n_folds: 5

noise_model:
  dof_correction: "sqrt(n/(n-2))"
  offset_strata: baseline_ms_tertile
  offset_constant_if_rel_diff_below: 0.20
  fitted_bins_db: [0, 5, 10, 15, 20, 25, 30, .inf]
  spatial_corr_threshold: 0.10   # ≥ 0.10 逐点噪声用指数衰减空间协方差；否则独立
  target_censor_frac: 0.0415     # 开发集实测删失比例（校准只用开发集）

simulator:
  template_baseline_method: tobit_left_censored_0db   # 修订 R1
  template_baseline_tobit:
    censor_point_db: 0.0
    sigma: "fixed = sqrt(SD_point(f_ols)^2 + SD_offset(tertile)^2)，取自 data/noise_model.json，不逐点估计"
    all_censored_rule: "该眼该点全部检查都删失时 MLE 发散，截距取下界 −10 dB"
    no_censored_rule: "该眼该点无删失时 Tobit 等于 OLS"
    baseline_time: first_visit   # 截距取第 1 次检查时刻 t=0
  template_baseline_clip_db: [-10, 35]
  interval_padding: dev_empirical_intervals_with_replacement   # sim_train 与 sim_test 都从开发集间隔补抽
  primary_k: 4
  k_versions: [3, 4, 5, 6]
  n_rep_sim_train: 10
  n_rep_sim_test: 20
  seed_sim_train: 20261019
  seed_sim_test: 20261119       # 与 sim_train 分开
  truth_scale_primary: observed_floor_0db   # max(f(t),0) 的 OLS 斜率
  truth_scale_secondary: latent             # f(t) 的 OLS 斜率，只作描述
  scenarios:
    S0: {name: 稳定加年龄下降, slope_all: -0.1, positive: false}
    S1: {name: 白内障样, step_drop_db_uniform: [1, 2], step_before_visit: "uniform{2..k}", slope_other: 0.0, positive: false}
    S2: {name: 簇内局灶进展, n_clusters: [1, 2], slopes: [-0.5, -1.0, -2.0], positive: true}
    S3: {name: 跨簇弓形缺损, n_clusters_spanned: [2, 3], slopes: [-1.0, -2.0], positive: true}
    S4: {name: 旁中心小暗点, max_ecc_deg: 10, n_points: [3, 4, 5], slopes: [-1.5, -3.0], positive: true}
    S5: {name: 弥漫进展, slopes: [-0.3, -0.6], positive: true}
    S6: {name: 后期加快, slope_early: -0.3, slope_late: -1.5, knot: time_midpoint, positive: true}
    S7: {name: 二次型加速, mean_slope_at_k4: -1.0, positive: true}

sim_calibration_tolerance:
  ms_resid_sd_rel: 0.10
  offset_sd_rel: 0.10
  point_resid_sd_rel_per_bin: 0.15
  ms_slope_sd_rel: 0.15
  ms_slope_mix_bins_db_per_yr: [-1.0, -0.3]   # < −1.0 → S2；−1.0 到 −0.3 → S5；≥ −0.3 → S0
  censor_frac_abs: 0.01
  max_tuning_rounds: 3

sim_test_seal:
  sealed: true                 # 解封日前任何脚本都不许读 sim_test；评估脚本必须检查此开关
  unseal_not_before_date: "2027-01-04"   # 第 12 周第一天
  sha256: "efae5bfacb41eaa4be15488722697f2f546bb8ee5e6fe156d5f3c9a63e2f97fe"  # R7：sim_test.npz；解封日前评估脚本只核对此哈希，不读文件内容

methods:
  B1_min_visits: 3
  EB: {sensitivity_strata: 3, prior_var: "max(0, var(b) - mean(s^2))", shrink_target: eye_cluster_mean}
  EB_GK:                       # 修订 R2：仅作描述性对照
    slope: per_point_ols_full_series
    shrink_target: training_eye_population_mean_slope_at_location
    prior_var: "per location: tau_p^2 = max(0, var_train(b_p) - mean_train(s_p^2))"
    weight: "w = tau_p^2 / (tau_p^2 + s^2)"
    fit_on: training_eyes_only
    role: descriptive_only
    in_fixed_sequence_tests: false
    reference: "Gwak & Kang, MICCAI 2026 OMIA (OMIA_015)"
  B4:
    random_effects: [eye_intercept, eye_slope, eye_cluster_intercept, eye_cluster_slope, visit_intercept]
    spatial_term: true
    spatial_impl_preferred: spaMM_matern_within_hemifield
    spatial_impl_fallback: car_neighbor_graph_le6deg
    fallback_if_est_hours_over: 24
    nonconvergence_steps: [diagonal_re_cov, car_neighbor_graph_le6deg]
  B5: {self_weight: 0.5, neighbor_weight_total: 0.5}
  PoPLR: {n_perm: 1000, one_sided: negative_slopes}
  on_estimation_failure: substitute_B1_and_report_rate

g1mst:
  d_model: 32
  n_layers: 2
  ff_width: 64
  dropout: 0.1
  fourier_freqs: 8
  approx_params: 30000
  neighbor_bias: learnable     # 同半视野且 ≤6° 的点对加 +b；跨水平中线 −inf
  mask_modes: [random_20pct_tokens, whole_visit, whole_location]
  mask_frac: 0.20
  loss: censored_gaussian_nll
  optimizer: AdamW
  lr: 1.0e-3
  weight_decay: 1.0e-2
  batch_size: 16
  max_epochs: 400
  early_stop_patience: 30
  train_data: real_dev_only    # 主模型不用模拟数据；G1-MST-sim 只作 SA7
  inference_K: 20
  inference_min_masked_per_token: 2
  seed_aggregation: mean_of_per_seed_metrics   # H1/H2/H3：先按种子算指标再平均；H4：外推值按种子平均
  large_version: {d_model: 64, n_layers: 3, approx_params: 200000}

se_calibration:
  method: "c = quantile(|z|, 0.95) / 1.96，在 sim_train 上按方法×层级估计，真值用观察尺度"

evaluation:
  primary_endpoint: "sim_test, k=4, 簇级斜率 RMSE（10 簇等权、8 情景等权），G1-MST − B4"
  test_order: [H1, H3, H4, H2]   # 固定顺序检验
  success_definition: "95% 区间完全落在 G1-MST 更好的一侧"
  n_bootstrap: 2000
  bootstrap_unit: subject
  ci_level: 0.95
  detection_specificity: 0.95
  detection_threshold_from: sim_train_S0_S1
  real_detection_n_shuffles: 200
  real_detection_fpr: 0.05
  h4_min_visits: 5
  h4_history_visits: 4
  h4_mae_weighting: eye_equal
```

---

## 6. Data decisions fixed before publication (using data structure only; no competing method run)

1. **Subject split** (`data/split.json`, seed 20261019): development set 101 subjects / 185 eyes, held-out set 43 subjects / 78 eyes; stratification = subject's baseline MS tertile × whether the subject has an eye with ≥5 examinations. Baseline MS distribution KS p = 0.89 (subject level); Mann–Whitney p for examinations per eye = 0.82.
2. **G1 location mapping** (`data/g1_locations.csv`): coordinates from `locmaps$g1` of the R package visualFields 1.0.7; the 61 GRAPE values are arranged in an outward spiral from the centre as in Figure 2b of the original paper, checked point by point against the first examination of the right eye of subject 8 (59/59), and independently checked against the left-eye example of Figure 2c (58/59). The blind-spot placeholders take the centre of the grey ellipse in Figure 2b, (15.1°, −1.3°) (pixels were linearly calibrated to degrees using the 15 complete black squares in the figure, and the centre of the bounding box of the grey pixels was taken). All three validations passed.
3. **B4 spatial term**: after removing the examination global offset, the mean residual correlation between neighbouring points in the same hemifield within ≤6° = **0.507** (other point pairs −0.035). Computation: all 263 eyes and 1115 examinations; per-eye, per-point OLS residuals multiplied by \( \sqrt{n/(n-2)} \), with the mean of the 59 points subtracted at each examination; for each pair of locations the Pearson correlation is computed across the 1115 examinations and then averaged within groups; "neighbouring" = distance ≤ 6° and not crossing the horizontal midline (\( y_i y_j \ge 0 \)), with the central point (0,0) counted as neighbouring points within ≤ 6° on both sides (58 pairs in total; 1653 other pairs). 0.507 > 0.10 → **B4 includes a spatial term**. The implementation is fixed in week 3 according to the rules of Section 2.4.
4. **Noise model** (`data/noise_model.json`, development set only):
   - The examination global offset SD in the three baseline-MS strata is 1.86 / 1.32 / 1.27 dB (relative difference 47% ≥ 20%) → **stratified**; simulation uses the deconvolved offset SDs 1.62 / 1.11 / 1.14 dB. Self-check: simulating with it and re-estimating with the same algorithm gives a global offset SD of 1.45 (real development set 1.50).
   - Pointwise noise: smoothing-spline SD as a function of fitted sensitivity; exponentially decaying spatial correlation within a hemifield (a = 1.00, λ = 5.7°; fitted by least squares of the pairwise correlation \( \rho(d) = a\,e^{-d/\lambda} \), with a restricted to [0, 1], over all strictly same-hemifield point pairs (excluding the central point) of the development-set point-specific residuals e after removing the global offset), no correlation across the horizontal midline.
   - This file was generated under the template definition preceding revision R1 (floor at 0 dB); R1 takes only the σ-related parameters from it, and the censoring proportion is re-checked with Tobit templates during the calibration of Section 3.5.
5. **Cluster definition**: the 10 clusters of the Octopus EyeSuite G program (Bürki & Monhart, European Ophthalmic Review 2007, Figure 1b) are adopted; the central point (0,0) does not belong to any cluster. Cluster numbering is defined by this study:

| Cluster | Name | No. of points | Points included (x°, y°), right-eye visual field coordinates, +x temporal |
|---|---|---:|---|
| 1 | S_central | 4 | (-2,2), (2,2), (4,4), (8,2) |
| 2 | S_paracentral_nasal | 7 | (-4,4), (-8,2), (-2,8), (2,8), (-14,4), (-20,4), (-26,4) |
| 3 | S_arcuate | 6 | (-8,8), (8,8), (-12,12), (-4,14), (4,14), (-20,12) |
| 4 | S_peripheral | 8 | (12,12), (-20,20), (-12,20), (-4,20), (4,20), (12,20), (-8,26), (8,26) |
| 5 | S_temporal | 4 | (20,20), (20,12), (22,4), (26,8) |
| 6 | I_central | 4 | (2,-2), (-2,-2), (4,-4), (8,-2) |
| 7 | I_paracentral_nasal | 9 | (-4,-4), (8,-8), (3,-9), (-3,-9), (-8,-8), (-8,-2), (-14,-4), (-20,-4), (-26,-4) |
| 8 | I_arcuate | 4 | (4,-14), (-4,-14), (-12,-12), (-20,-12) |
| 9 | I_peripheral | 8 | (12,-12), (12,-20), (4,-20), (-4,-20), (-12,-20), (-20,-20), (8,-26), (-8,-26) |
| 10 | I_temporal | 4 | (22,-4), (20,-12), (20,-20), (26,-8) |

6. **SHA-256 of fixed files** (computed 2026-10-20):

| File | SHA-256 |
|---|---|
| `data/split.json` | `5e960c3ede4b8334017f0b6981a01b8d6e87a013a89026d927d0f6d1b2361467` |
| `data/g1_locations.csv` | `73d64f945f8645eb3c98b7db54cdf9111a014e37d175871ac55be348d6cd03ea` |
| `data/noise_model.json` | `fcd6b649bbf3eceb5c5fc6d22f580e24883a466134c4fa18669f8c273c0b56a6` |

---

## 7. Revision log

| ID | Date (Asia/Shanghai) | Content | Decided by | Before any model results? |
|---|---|---|---|---|
| — | 2026-10-20 approx. 12:11 | Initial draft | Drafted by the executing agent | Yes |
| R1 | 2026-10-20 12:39 | Simulation template baseline switched to the pointwise Tobit (left-censored at 0 dB) intercept, allowing < 0 dB, range −10–35 dB; calibration carried out as usual | User | Yes (no simulated data had yet been generated) |
| R2 | 2026-10-20 12:39 | Added the descriptive comparator EB-GK, not entering the fixed-sequence tests H1–H4 | User | Yes (no method had yet been run) |
| R3 | 2026-10-20 approx. 16:35 | Data-step files (split, locations, noise model) regenerated before publication; the numbers and hashes in Part 6 and the censoring target updated accordingly | User | Yes |
| R4 | 2026-10-20 | At 16:39 (UTC+8) on 2026-10-20, an earlier, superseded draft of this preregistration (commit 3129355) was pushed directly to the main branch of this repository and was briefly public. That draft is not this preregistration; its numbers, hashes and code do not apply. This commit replaces the draft in full with the present version; the draft remains in the git history for inspection. This revision does not change any test, endpoint, data, or the numbers and hashes in section 6. | User | Yes |
| R5 | 2026-10-20 | As announced in section 3.4, appended before any simulated data are generated: the point-set generation rules for the S3 arc and the S4 scotoma, the implementation details of the simulator that the text left open, and the SHA-256 of every simulator source file (section 7.1). No endpoint, testing order, decision rule, scenario parameter, or number or hash in section 6 is changed | User | Yes (no simulated data generated, no competing method run) |
| R6 | 2026-10-20 | Corrected the status line at the top: it read "draft v2, unpublished", but v2 was published on 2026-10-20 (GitHub tag `v2.0-prereg`, Zenodo DOI 10.5281/zenodo.23494626); R5 and R6 are later revisions. Only that line of text changes; no content changes | User | Yes (no simulated data generated, no competing method run) |
| R7 | 2026-10-21 | As announced in section 3.4 and rule C-3 of revision R5, appended immediately after calibration finished and `sim_test` was generated: the SHA-256 of `data/sim_test.npz`, the final κ_offset / κ_point, the SHA-256 of `data/sim_calibration.json`, and a summary of the five calibration checks (section 7.2). No endpoint, testing order, decision rule, scenario parameter, or number or hash in section 6 is changed; `sim_test` is not read for any method evaluation | User | Yes (no competing method run; `sim_test` remains sealed until 2027-01-04) |
| R8 | 2026-10-21 | B4 spatial term locked to spaMM Matérn (within hemifield). Formal timing: random 20 development eyes (seed 20261021); spaMM fit took 30.90 s; linear extrapolation to all 185 development training eyes ≈ 0.0794 h ≤ 24 h, so `spaMM_matern_within_hemifield` is adopted (CAR not used). Implementation in `src/methods/lmm.py` (`spatial_impl` and Matérn parameters); timing script `src/17b_b4_timing_formal.py`; SHA-256 in section 7.3. **Disclosure**: the preregistration announced the week-3 window as 2026-11-02 to 2026-11-08; at the investigator's request, timing and this revision were completed early on 2026-10-21. No endpoint, testing order, decision rule, scenario parameter, or section-6 number/hash is changed; `sim_test` is not read. | User | Yes |
| R9 | 2026-10-21 | Week-5 checkpoint (§3.5): on `sim_train` (k = 4, scenario-equal-weight cluster RMSE) compare B4 (`spaMM_matern_within_hemifield`) vs B1; 2000 subject-bootstrap resamples of development-set template persons. Result: RMSE_B4 = 0.926384, RMSE_B1 = 1.379045, RMSE_B4 − RMSE_B1 = −0.452661, 95% CI [−0.493479, −0.405430], entirely < 0 → **pass**. **Disclosure**: announced window was 2026-11-16 to 2026-11-22; run early on 2026-10-21 at the investigator's request. First implementation failed due to overshrinkage and Matérn kriging toward the visit mean; after correcting residual Matérn / variance components / EB-BLUP **without** changing the R8-locked string `spaMM_matern_within_hemifield`, the checkpoint was re-run and passed. Updated `src/methods/lmm.py` SHA-256 = `69b7293bb6393cb19a534b3c7d53b6bf99310d3c2afb758fd22e32c603aff477` (section 7.4). Endpoints, test order, decision rules, scenario parameters, section-6 hashes and the R8 `spatial_impl` string are unchanged; `sim_test` was not read. | User | No (this revision records the §3.5 checkpoint outcome) |
| R10 | 2026-10-21 | Fix B4 point-level SEs: inference previously broadcast the cluster trajectory onto every point in the cluster, forcing pointwise residuals to zero and collapsing `se_point`; the denoised surface is now the within-hemifield residual-Matérn smooth, and pointwise OLS/SEs are computed on that surface. Cluster slopes/SEs and the R9 week-5 checkpoint cluster RMSEs (B4 = 0.926384, B1 = 1.379045) are **unchanged**. Updated `src/methods/lmm.py` SHA-256 = `0bc51e64a750445755ed099b9ac567f0a60344876b5f9eae03171ef74e8a2817` (section 7.5; supersedes `69b7293bb6393cb19a534b3c7d53b6bf99310d3c2afb758fd22e32c603aff477` in R9). **Does not** change the R8-locked string `spaMM_matern_within_hemifield`; endpoints, test order, decision rules, scenario parameters and section-6 hashes are unchanged; `sim_test` was not read. | User | No (implementation fix; cluster primary-endpoint numbers unchanged) |

Any change after publication is appended as a new row (with date, content and reason) and reported item by item in the paper. Announced follow-up revisions: the B4 spatial-term choice was fulfilled by R8 (timing completed early on 2026-10-21; see section 7.3); the week-5 B4 vs B1 checkpoint was fulfilled by R9 (run early on 2026-10-21 and passed; see section 7.4); the B4 point-SE fix and implementation-hash update were fulfilled by R10 (see section 7.5); S3/S4 point-set rules and simulator code hashes, and the `sim_test` SHA-256, were fulfilled by R5 and R7 respectively.

### 7.1 Details of revision R5: S3/S4 point-set generation rules, open simulator details, and code SHA-256

This section was fixed before any simulated data were generated. Rule identifiers match the source-code comments. All rules use only the public G1 coordinates and the 10-cluster definition; no participant data were used.

**S3 arc (S7 uses the same rule)**
- S3-1 Nerve fibre model: Jansonius et al. 2009 (Vision Res 49:2157), mean parameters (β_s = −1.9, β_i = 0.7, r₀ = 4°, disc centre at (15°, 2°), modified polar coordinates of Appendix A). For the superior retina c = 1.9 + 1.4·tanh((φ₀ − 121)/14), the sign that matches the paper's statement that c increases from the nasal to the temporal side. Visual field → retina: x unchanged, y negated.
- S3-2 Each non-central point gets an entry angle φ₀: the grid value (step 0.001°) on |φ₀| ∈ [60°, 180°] with the smallest residual; a solution exists if |residual| < 0.05°. Points without a solution and the central point do not enter S3/S7. The points without a solution are clusters 5 and 10 in the temporal field: they lie nasal to the disc, where fibres enter roughly radially and the model does not apply.
- S3-3 Candidate arcs: within one hemifield, for each point i with a solution take the window {j : φ₀ᵢ ≤ |φ₀ⱼ| < φ₀ᵢ + 20°}. Keep and de-duplicate the sets that satisfy all of the following: at least 3 points; they touch 2 or 3 clusters; they cover only part of each touched cluster; at least one point has eccentricity > 10°, which separates S3 from the central 10° of S4. Result: 22 arcs (15 spanning 2 clusters, 7 spanning 3; 15 superior, 7 inferior), each with 4–8 points. The full list is in `results/sim_pointsets.json` and in the table below.
- S3-4 Sampling: draw the number of clusters spanned from {2, 3} with equal probability, then one candidate uniformly among those with that number (both hemifields pooled). The slope level is drawn from {−1, −2} dB/year with equal probability; all points of the arc share it.

| ID | Hemifield | Clusters spanned (clusters) | Points (loc59) |
|---|---|---|---|
| 0 | inferior | 2 (7, 8) | 12, 13, 14, 24, 25, 40 |
| 1 | inferior | 2 (7, 8) | 12, 13, 14, 24, 25, 40, 54 |
| 2 | inferior | 2 (7, 8) | 12, 13, 23, 24, 25, 39, 40, 54 |
| 3 | inferior | 2 (7, 8) | 12, 13, 23, 24, 25, 40, 54 |
| 4 | inferior | 2 (8, 9) | 21, 22, 35, 36, 37, 38, 53 |
| 5 | inferior | 3 (7, 8, 9) | 22, 23, 36, 37, 38, 39, 53, 54 |
| 6 | inferior | 2 (8, 9) | 22, 35, 36, 37, 38, 53 |
| 7 | superior | 2 (2, 3) | 15, 16, 17, 18, 19 |
| 8 | superior | 2 (2, 3) | 15, 17, 18, 19 |
| 9 | superior | 2 (2, 3) | 16, 17, 18, 19, 26 |
| 10 | superior | 2 (2, 3) | 16, 19, 26, 27, 28, 29, 41 |
| 11 | superior | 2 (2, 3) | 16, 19, 26, 28 |
| 12 | superior | 2 (2, 3) | 16, 19, 26, 28, 29, 41 |
| 13 | superior | 2 (2, 3) | 16, 19, 26, 28, 41 |
| 14 | superior | 3 (2, 3, 4) | 27, 28, 29, 41, 42, 44, 45, 55 |
| 15 | superior | 3 (2, 3, 4) | 27, 28, 29, 41, 42, 45, 55 |
| 16 | superior | 2 (2, 3) | 27, 28, 29, 41, 42, 55 |
| 17 | superior | 2 (2, 3) | 27, 28, 29, 41, 55 |
| 18 | superior | 3 (2, 3, 4) | 27, 29, 42, 43, 44, 45, 55 |
| 19 | superior | 3 (2, 3, 4) | 30, 42, 43, 44, 45, 46, 55 |
| 20 | superior | 3 (2, 3, 4) | 30, 42, 43, 44, 45, 46, 55, 56 |
| 21 | superior | 3 (2, 3, 4) | 30, 42, 43, 44, 45, 55 |

**S4 paracentral scotoma**
- S4-1 Eligible points: the 17 points with eccentricity √(x² + y²) ≤ 10°, including the central point.
- S4-2 Candidate sets: n eligible points that are connected in the neighbour graph (NB-1 below) and whose non-central points all lie in one hemifield. There are 56, 97 and 133 sets for n = 3, 4 and 5.
- S4-3 Sampling: n from {3, 4, 5} with equal probability, then one set uniformly among those of size n. The slope level is drawn from {−1.5, −3} dB/year with equal probability; all points of the set share it.

**Other scenarios**
- U-1 Unaffected points in S3, S4 and S7 follow S0 (−0.1 dB/year), the same rule the text states for S2 and S6. In every focal scenario, the slope of an affected point is the drawn value itself; −0.1 is not added to it.
- S2-1 S2: the number of clusters is drawn from {1, 2} with equal probability; the clusters are drawn uniformly without replacement from the 10. A series draws one slope level, shared by all of its selected clusters.
- S6-1 S6: locations are drawn independently with the S2 rule (not shared with the S2 series of the same index). The knot is the time midpoint (t₁ + t_k)/2 of the observation window of each k version, so it depends on k.
- S7-1 S7: locations are drawn independently with the S3 rule. "Mean slope −1 dB/year at the end of 4 examinations" is defined as an OLS slope of −1 for the latent trajectory −a t² over the first 4 examination times of the series. That is, a = 1 / OLS slope(t²); for equally spaced visits this equals the end-to-end slope.
- S7-2 k ≠ 4: a is set from the first 4 examination times only, so all k versions share one trajectory. k = 3 uses the first 3 examinations; k = 5 and 6 keep accelerating, so their true slope is steeper than −1.
- S1-1 S1: each series draws one u ~ U(0, 1). In the k version the drop occurs before examination j_k = 2 + ⌊u(k − 1)⌋, which is uniform on {2, …, k}.
- NB-1 Neighbour convention for the central point: the convention of section 6 item 3 is the only definition in the document. Under it, "neighbours" are points at distance ≤ 6° with yᵢyⱼ ≥ 0, so the central point neighbours every point within 6° in both hemifields. It applies to every use of "neighbouring points" in the document: S4, B5 smoothing and the G1-MST attention bias. For noise generation the central point still joins the superior block, per section 3.3.

**Templates and noise**
- T-1 Tobit σ: each eye and point gets one constant σₚ = √( mean_v SD_point(f̂_OLS,v)² + SD_offset² ), with the mean taken over all of that eye's examinations. A σ that varied by examination would make the uncensored MLE weighted least squares rather than OLS, contradicting R1's "equals OLS when there is no censoring". A constant σ makes the two agree.
- T-2 SD_offset is the deconvolved, stratified offset SD from `noise_model.json` (the value used for simulation), not multiplied by the section 3.5 calibration factors. Templates are computed once and are not re-estimated during calibration.
- T-3 With partial censoring, L-BFGS-B starts from OLS and searches a ∈ [−100, 100] dB and b ∈ [−30, 30] dB/year (to prevent MLE divergence); a is then clipped to [−10, 35] dB. Templates use all of the eye's examinations.
- N-1 Pointwise noise SD is evaluated at the latent true value f(t_v), clipped at ≥ 0 per the `noise_model.json` rule. y* = f + δ_v + κ_point·SD_point(f)·(L z), with δ_v ~ N(0, (κ_offset·SD_offset)²) and L the Cholesky factor of ρ(d) = a·e^(−d/λ) within hemifield blocks. Where y* < 0 the value is recorded as 0 dB with `censored = True`.
- N-2 Baseline-MS tertiles: the cut points are the tertiles of baseline MS over development-set eyes (as in `03_noise_model.py`); holdout template eyes use the development cut points.
- N-3 Simulated values are neither rounded nor capped. The text specifies only censoring at 0 dB.

**Schedules, random numbers and output**
- K-1 Nested schedules: each series first builds a 6-examination schedule. It takes the template's first min(n, 6) real times and fills any shortfall by drawing, with replacement, from the pool of adjacent intervals of all development eyes. The k version uses the first k examinations, and all k versions share the same standard-normal draws. Padded examinations have missing IOP; age = template baseline age + t.
- R-1 Random numbers: each series derives 3 independent streams (schedule, scenario parameters, noise) from `SeedSequence([seed, template index (eye_id ascending), scenario index, replicate])`. Seeds come from `config.yaml` (`sim_train` 20261019, `sim_test` 20261119). Results do not depend on generation order. SA8 (no exam-level offset) and SA10 (spatially independent pointwise noise) use the same streams, so they are paired with the main version.
- O-1 Output: `data/sim_train.npz` and `data/sim_test.npz`; sensitivity variants get the suffix `_SA8` or `_SA10`. Each k has one group of arrays: times, observations, censoring, latent trajectory, IOP, age, and pointwise, cluster and global truths on both scales. Scenario parameters are stored alongside. Zip timestamps and member order are fixed, so identical inputs give identical bytes.

**Calibration (section 3.5)**
- C-1 The simulated side uses the k = 4 version of `sim_train`, with all five statistics weighted by the S0/S5/S2 mixture weights (the text states the mixture only for the slope SD). The real side uses all development-set examinations, with the same algorithm as `03_noise_model.py`.
- C-2 Only two multiplicative factors are tuned, κ_offset and κ_point. Each round updates them by the real/simulated ratio of the exam-level offset SD and of the pooled pointwise SD. Scenarios, templates and spatial-correlation parameters are not tuned; at most 3 rounds. The observed MS slope SD and the censored fraction cannot be tuned directly; any remaining gap after tuning is reported in the paper.
- C-3 Order: publish this revision → generate `sim_train` → calibrate, regenerating `sim_train` with the same seed in each round → generate `sim_test` → immediately append a revision recording the SHA-256 of `sim_test`, the final κ values and the SHA-256 of `data/sim_calibration.json`.

**Guards**
- G-1 `04_simulator.py` refuses to generate anything, and writes no file, in any of these cases: `config.yaml` revisions lack R5; a listed source hash does not match the file on disk; calibration is not final when `sim_test` is requested; the `sim_test` file already exists. The script never reads `sim_test`.

**SHA-256 of the simulator source and point-set files (computed 2026-10-20)**

| File | SHA-256 |
|---|---|
| `src/sim_core.py` | `559b1f87ee7d3350941db352c2133deb6293ee467da9b9d0a950152b6ab094cc` |
| `src/04_simulator.py` | `e336cf9b5475434b5b5fb02ea1a6d1bffb098456be91d714aeae84c1f5d629ec` |
| `src/05_calibrate_sim.py` | `a9187a926fcef90f2ec3ebe6c1110a992375119c1bd684acc18fdd1f430e94c2` |
| `src/test_simulator_toy.py` | `81e9f1863cf9600348f2edd2f3fb3efbb5fa26f368feab5f89e88c88499f6a8d` |
| `results/sim_pointsets.json` | `f9fbce6aa825df59448f7be6273afbd4eba4784acc3bc1653b34d9be660b6a0e` |

Before this revision, only two things were run. One is `04_simulator.py --pointsets-only`, which reads only the public `data/g1_locations.csv` and writes `results/sim_pointsets.json`. The other is `test_simulator_toy.py`, which uses the public point geometry together with a fixed toy noise model, toy templates and toy participant data, all unrelated to GRAPE; the test's outputs go to a temporary directory that is deleted at the end of the test. No simulated data were generated from GRAPE-derived templates or noise models, and no Tobit template was computed for any GRAPE eye.

### 7.2 Details of revision R7: sealed `sim_test` hash and final calibration coefficients

Per rule C-3 of revision R5, `sim_test` was generated after calibration reached `final = true`, and the hashes and coefficients below are recorded immediately. `sim_test` remains sealed until 2027-01-04 (`sim_test_seal.sealed = true` in `config.yaml`); this revision does not authorise any method to read it.

**Sealed and calibration file SHA-256 (computed 2026-10-21, Asia/Shanghai)**

| File | SHA-256 |
|---|---|
| `data/sim_test.npz` | `efae5bfacb41eaa4be15488722697f2f546bb8ee5e6fe156d5f3c9a63e2f97fe` |
| `data/sim_calibration.json` | `22f615bdf64bc321dcc40b067d82f7ad112faa50fa1dd1bf39bc6f3973361350` |
| `data/sim_train.npz` (final round, for audit; not sealed) | `2bc74625ec6ca36f049ef293f99e166e71ed6ffaa083503bd7240c0ead3c7a3a` |

**Final calibration coefficients (written to `data/sim_calibration.json` and used to generate the `sim_train` / `sim_test` above)**

- κ_offset = 1.0643078763377154
- κ_point = 1.0518576611587154
- `final = true`, `round = 3` (hit the maximum of 3 tuning rounds under rule C-2 of revision R5)
- Generation parameters: `sim_train` seed 20261019, n_rep = 10, 185 development-set template eyes; `sim_test` seed 20261119, n_rep = 20, 78 holdout template eyes; both use the final κ above; numpy 2.5.4

**Five calibration checks (final round; `sim_train` k = 4; S0/S5/S2 mix weights 0.497 / 0.303 / 0.200)**

| Statistic | Real (development set) | Simulated | Relative / absolute gap | Tolerance | Pass |
|---|---|---|---|---|---|
| MS residual SD | 1.5016 | 1.5014 | rel 0.01% | ≤ 10% | Yes |
| Visit-level offset SD | 1.5016 | 1.5014 | rel 0.01% | ≤ 10% | Yes |
| Pointwise residual SD (all 7 bins) | — | — | max bin rel ≈ 14.5% (bin 0) | ≤ 15% per bin | Yes |
| Observed MS slope SD | 1.3886 | 0.9224 | rel 33.6% | ≤ 15% | **No** |
| Censoring fraction | target 0.0415; sim 0.0466 | — | abs 0.51 percentage points | ≤ 1 pp | Yes |

Per rule C-2: only κ_offset and κ_point are tuned; observed MS slope SD and the censoring fraction cannot be tuned directly, and any remaining gap after 3 rounds is reported in the text. Only observed MS slope SD missed its tolerance (simulated side narrower); the gap is recorded as-is. Scenario parameters, templates and mix weights are not changed, and calibration is not reopened. The other four of the five checks all passed at the final κ.

This revision also sets `sim_test_seal.sha256` in `config.yaml` from `null` to the `sim_test.npz` hash above, and appends R7 to `revisions`. No hash in section 6 and no simulator source file is changed.

### 7.3 Details of revision R8: B4 spatial-term implementation (week-3 timing completed early)

Per the fixed rules in section 2.4, spaMM Matérn variance-component estimation (within hemifield; separate fits above/below the horizontal meridian) was timed on a random sample of 20 development eyes and linearly extrapolated by eye count to all development training eyes.

**Timing results (2026-10-21, Asia/Shanghai)**

| Item | Value |
|---|---|
| Random seed | 20261021 |
| Eyes timed | 20 |
| Development training eyes (total) | 185 |
| spaMM fit time | 30.9003 s (wall 32.02 s) |
| Extrapolation to all development training eyes | 0.079397 h |
| 24-hour threshold | not exceeded → **adopt spaMM Matérn** |
| Chosen implementation | `spaMM_matern_within_hemifield` |
| Timing record | `results/baselines/b4_timing_week3_formal.json` (local; need not be in the public repo) |

Formula (each hemifield): `y ~ 1 + t + (1+t|eye) + (1+t|eye_cl) + (1|eye_vis) + Matern(1|x+yy)`, REML, R `spaMM` 4.7.17.

**Implementation SHA-256 (2026-10-21)**

| File | SHA-256 |
|---|---|
| `src/methods/lmm.py` | `1a5f8f23e129a462a907b8016a2ec7f197d74556cb53629ad02c58e8c94d5a12` |
| `src/17b_b4_timing_formal.py` | `6b4e76fd552265ad915fe83f092746dfdfbe65972eeaa6d5d8a091e73fd09227` |

At inference: hierarchical variance components and Matérn parameters are locked on training eyes (ν = 0.5; ρ aligned so correlation at 6° matches the development-set neighbour residual correlation; partial sill and nugget from residual decomposition); for a new eye, parameters are held fixed, each visit is Matérn-GLS-smoothed within hemifield, then cluster-level two-level BLUP is applied. `B4.spatial_impl` is the fixed string `spaMM_matern_within_hemifield`.

**Disclosure of early completion**: section 2.4 originally scheduled timing for week 3 (2026-11-02 to 2026-11-08). At the investigator's request, formal timing and this revision were completed and published on **2026-10-21**. The rule itself (20 eyes, linear extrapolation, ≤ 24 h → spaMM else CAR) is unchanged. The B4 spatial implementation is not changed thereafter.

This revision appends R8 to `revisions` in `config.yaml` and records the chosen implementation above. H1–H4, scenarios, truth scale, sealing rules and section-6 hashes are unchanged.

### 7.4 Details of revision R9: week-5 checkpoint (run early; passed) and updated B4 implementation hash

Per §3.5, B4 was compared with B1 on `sim_train` (k = 4, scenario-equal-weight cluster-slope RMSE); the bootstrap unit is the development-set template person (`meta_template_subject`), 2000 resamples.

**Checkpoint results (2026-10-21, Asia/Shanghai; after the implementation fix)**

| Statistic | Value |
|---|---|
| RMSE_B4 | 0.926384 |
| RMSE_B1 | 1.379045 |
| RMSE_B4 − RMSE_B1 | −0.452661 |
| 95% CI | [−0.493479, −0.405430] |
| Decision | **pass** (interval entirely < 0) |
| `spatial_impl` | `spaMM_matern_within_hemifield` (same string locked by R8) |
| Record | `results/baselines/b4_vs_b1_checkpoint.json` (SHA-256 `a93624977e71083c0129ba4486f24339b603ad7faccdc488155624a53badc2ba`) |
| Diagnostic summary | `results/baselines/b4_debug.md` (SHA-256 `e50a4935d338ae204a84c49f2055b87c4e9c7dc54330c3782a7e054066b941e4`) |

**Disclosure of early run**: §3.5 originally scheduled the checkpoint for week 5 (2026-11-16 to 2026-11-22). At the investigator's request it was run on **2026-10-21**. The first run failed (B4 worse than B1); the root cause was implementation-level overshrinkage and kriging toward the visit mean (see diagnostic summary), **not** an error in the R8-locked spatial-term definition string. After the fix, the same protocol was re-run and passed.

**Implementation correction (locked `spatial_impl` string unchanged)**: within-hemifield Matérn smoothing is applied to **residuals** relative to training point means; variance components are re-estimated on the smoothed cluster trajectories; at prediction, cluster OLS is EB/BLUP-shrunk toward the population slope. The timing rule and chosen implementation string in section 7.3 (R8) are unchanged; this revision only updates the implementation-file hash and records the checkpoint.

**Implementation SHA-256 (2026-10-21, this revision)**

| File | SHA-256 |
|---|---|
| `src/methods/lmm.py` | `69b7293bb6393cb19a534b3c7d53b6bf99310d3c2afb758fd22e32c603aff477` |
| `src/18_b4_vs_b1_checkpoint.py` | `6766e4f2b47dc79b30809ee86406bf6b5b91b757a1b59d35ceb263e8443f8d25` |

(The `lmm.py` hash in the R8 section 7.3 table is the pre-fix version; this table supersedes it.)

This revision appends R9 to `revisions` in `config.yaml`. H1–H4, scenarios, truth scale, sealing rules, section-6 hashes and the R8-locked string `spaMM_matern_within_hemifield` are unchanged; `sim_test` was not read.

### 7.5 Details of revision R10: B4 point-SE fix and updated implementation hash

On the R9-locked `spaMM_matern_within_hemifield` implementation, inference had been broadcasting the cluster trajectory onto every point in the cluster, so pointwise OLS residuals were identically zero and `se_point` collapsed (point-level SE calibration blew up). This revision changes only that:

- `denoised` is the within-hemifield residual-Matérn-smoothed surface (relative to training point means);
- cluster slopes / SEs remain those written by EB-BLUP, identical to the cluster estimates used for the R9 checkpoint;
- the `spatial_impl` string is **unchanged**.

**Implementation SHA-256 (2026-10-21, this revision)**

| File | SHA-256 |
|---|---|
| `src/methods/lmm.py` | `0bc51e64a750445755ed099b9ac567f0a60344876b5f9eae03171ef74e8a2817` |

(The `lmm.py` hash in the R9 section 7.4 table, `69b7293bb6393cb19a534b3c7d53b6bf99310d3c2afb758fd22e32c603aff477`, is the pre–point-SE-fix version; this table supersedes it.)

**Cluster numbers unchanged**: the week-5 checkpoint (R9) scenario-equal-weight cluster RMSEs remain B4 = 0.926384, B1 = 1.379045, with 95% CI entirely < 0. This revision does not re-run or rewrite that checkpoint decision.

This revision appends R10 to `revisions` in `config.yaml`. H1–H4, scenarios, truth scale, sealing rules, section-6 hashes and the R8-locked string `spaMM_matern_within_hemifield` are unchanged; `sim_test` was not read.


