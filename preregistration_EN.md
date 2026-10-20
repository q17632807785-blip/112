# Preregistration: A Specificity-Aligned Benchmark of Denoising Methods for Public Longitudinal Octopus G1 Visual Fields (GRAPE, Direction A)

> **Status: draft v2, not yet published.** To be published by the investigator in the public GitHub repository https://github.com/q17632807785-blip/112: this file is committed, the commit is tagged and a release is created, and the release is archived on Zenodo to obtain a DOI (GitHub commit timestamps can be rewritten; the Zenodo archive provides an immutable timestamp). The repository does not contain the raw GRAPE data or any per-subject derived data (`data/*.parquet`, `split.json`); it contains only this file, `config.yaml`, `g1_locations.csv`, and code. Section 6 lists the SHA-256 hashes of the files that are not made public.
> Date of generation: 2026-10-20 (Asia/Shanghai). Basis: `方向A_研究计划_最终版.md` (Direction A research plan, final version; hereafter "the plan") and Revisions R1 and R2 of 2026-10-20.
> This file supersedes `preregistration_draft.md`. Before publication, `config.yaml` must be synchronised verbatim with Section 5.
> *Translation note:* this is an English translation of the Chinese preregistration `preregistration.md`. The Chinese version is authoritative. All numbers and hashes are identical. The YAML block in Section 5 is reproduced verbatim from `config.yaml`; its comments and some string values are in Chinese (scenario names are translated in the prose table of Section 3.4).

## 0. Title, data, positioning

- Data: GRAPE (Huang et al., Sci Data 2023;10:520), 263 eyes / 144 subjects / 1115 Octopus 900 G1 visual field examinations (Second Affiliated Hospital of Zhejiang University School of Medicine).
- Positioning:

> On the only public longitudinal Octopus G1 dataset, we establish a specificity-aligned benchmark of visual field denoising methods whose primary endpoint is slope error against simulated ground truth, comparing statistical models with a masked denoising model that handles 0 dB censoring, examination-level global shifts, and spatiotemporal correlation.

- Additional related work: Gwak & Kang, "When to Denoise Longitudinal Visual Field Labels: Anchor-Preserving Supervision under Short Follow-up", MICCAI 2026 OMIA Workshop (open-access full text: https://papers.miccai.org/miccai-2026-sat/paper/OMIA_015.pdf). On the 263 GRAPE eyes, that work applied **empirical Bayes shrinkage to follow-up pointwise slopes (shrinking toward the per-location population mean of the training eyes)**, solely to provide training labels for a ResNet-50 that maps a single fundus photograph to the first visual field; with an eye-level split and three visits, MAE improved slightly by 0.075 dB, and the gain disappeared under a subject-level split. It did not assess slope accuracy, simulated ground truth, specificity-aligned detection, or censoring. The EB comparator of the present study (shrinkage toward the same-cluster mean of the eye) differs from that shrinkage target; an EB-GK variant is additionally included as a descriptive comparator (Section 2.2).

### 0.1 Prior knowledge of the data before publication (disclosure)

The investigators had access to the GRAPE data before publishing this preregistration, as detailed below. Apart from this, no competing method has been run on simulated data, and no comparison between methods has been produced.

1. **The v2 project** (2026-10-20 00:39–03:41): on the same data, linear mixed models were used to test whether the fundus photograph at the first visit explains the rate of visual field decline (244-eye cohort); the primary test gave a subject-level permutation p = 0.085. That project computed the MS residual SD, slope reliability, and the signal ceiling.
2. **Data facts** (Section 1 of the plan): summary statistics were computed on **all 263 eyes**, including the distribution of number of examinations, the censoring fraction, MS residual SD, slope SD and standard error, pointwise residual SD, examination-level global shift SD, and inter-point residual correlation. These computations preceded the population split, so subjects in the hold-out set contributed to these numbers in aggregate form; no individual-eye results of the hold-out set were examined.
3. **Phase-1 data steps** (2026-10-20 12:01–12:11): construction of the long table, G1 location mapping and its three validation checks (the neighbour residual correlation was computed on all examinations), the population split, and the noise model (development set only). The noise-model script includes a self-check simulation: data generated from the noise model are used to re-estimate the global shift SD and the censoring fraction (results in Section 6, item 4); this concerns only the noise model itself, and no competing method was run.
4. **Pilot run by another execution agent (`grape_pilot`)**: another execution agent (view1) working for the same user ran a pilot on the GRAPE data in a `grape_pilot` directory in its execution environment, from approx. 11:50 to 13:33 Beijing time on 2026-10-20 (last computational output 13:31; interrupted and resumed approx. 12:47–12:53); two summary documents were subsequently written at 15:25 and 15:34:45. The pilot went beyond checking that the code runs: judging from file names only, it includes G1-MST training and inference, a B4 variance-component step, and evaluation summaries. Its code is not the implementation described in this preregistration.
   - **Sealing**: the whole directory was sealed at 15:46 (`grape_pilot_sealed.tar.gz`, 29,857 files, SHA-256 `a8873d3e9335bcede283d3d879ef34a189edc7208dcc45dfee5b4ffaaa0cb24f`, per-file list `grape_pilot_manifest.sha256`); copies outside the directory are recorded by SHA-256 in `sealed/external_copies.sha256`; an earlier seal at 13:11 was superseded because the directory was written to after it, and is kept unchanged (see Revisions S1 and S2).
   - **Timing**: draft v2 of this preregistration (including the hypotheses, testing order, and analysis plan) was delivered to the execution agent at 13:05:22 on 2026-10-20 (basis: modification time of the attachment file; the work log records receipt at 13:06–13:07). Subsequent changes were limited to synchronising `config.yaml` with Section 5, filling in hashes, the English translation, and this disclosure. According to file modification times, output files appeared in the pilot directory from 11:51 onward, i.e. before v2 was delivered; its last computational output (13:31) postdates the delivery of v2.
   - **Exposure**: according to the investigator, the investigator has not read any pilot outputs. The execution agent opened `pilot_report.md` (modified 11:59:33) once, at approx. 12:40–13:05 (time not precisely recorded), to describe the pilot in the work log; the log records only that "the pipeline runs end to end; numbers are not interpreted". No other file in the directory or in the external copies has been opened since, and sealing involved listing only file names, sizes, and hashes. The initial draft of this preregistration (approx. 12:11) and Revisions R1 and R2 (12:39) predate this access.
   - **Use**: the investigator decided at 15:47 not to view or use any pilot results, and reaffirmed this at 15:57 and 16:03; according to the investigator, no content of this preregistration was changed on the basis of pilot results.

---

## 1. Research questions, endpoints, testing order

### 1.1 Primary endpoint H1

**H1**: On `sim_test`, with slopes estimated from exactly 4 examinations for every simulated series, the **root mean square error of cluster-level slopes** (10 clusters, dB/year) of G1-MST is lower than that of B4.

- **Computation of RMSE**: for each scenario \(s\), \( \mathrm{MSE}_s = \) the mean of \( (\hat b - b_{true})^2 \) over all series × 10 clusters of that scenario (the 10 clusters are weighted equally, not by number of points); the 8 scenarios are combined with equal weights: \( \mathrm{RMSE} = \sqrt{\tfrac18 \sum_s \mathrm{MSE}_s} \). \( \sqrt{\mathrm{MSE}_s} \) is also reported per scenario.
- **Seeds**: for G1-MST, the RMSE is computed separately for each of the 5 seeds and the 5 RMSEs are averaged (the estimates of the 5 seeds are not first averaged into an ensemble). Methods that do not require training seeds are computed once.
- **Statistic**: the difference \( \Delta = \mathrm{RMSE}_{G1\text{-}MST} - \mathrm{RMSE}_{B4} \).
- **Interval**: the subjects of the hold-out set (43 subjects) are resampled with replacement 2000 times; each resample carries all series generated from all template eyes of the sampled subjects, and \( \Delta \) is recomputed; the 2.5% and 97.5% quantiles form the 95% interval.
- **Decision**: if the upper limit of the interval is < 0, G1-MST is better (H1 holds); if the interval contains 0, the two cannot be distinguished; if the lower limit is > 0, B4 is better. The latter two outcomes both count as H1 not holding.

### 1.2 Key secondary endpoints and fixed testing order

The testing order is fixed as **H1 → H3 → H4 → H2**. "Holds" always means that G1-MST is superior to B4 (the 95% interval lies entirely on the side favouring G1-MST). A hypothesis is formally tested only if the preceding one holds; once a hypothesis does not hold, the subsequent ones are still computed and reported, but only descriptively, with no claim of superiority. Each uses a two-sided 95% interval (equivalent to one-sided α = 0.025), and the fixed sequence keeps the overall type I error rate at or below 0.025 (one-sided).

- **H3**: on `sim_test`, at k = 4, detection sensitivity at 95% specificity for the focal progression scenarios (S2, S3, S4, S6, S7 combined), G1-MST versus B4. The decision statistic, threshold, and interval algorithm are given in Section 3.6.
- **H4 (real-data anchor)**: real eyes with at least 5 examinations (88 eyes); cluster-level trends estimated from the first 4 examinations are extrapolated to predict the observed cluster means of the 5th and later examinations, and MAE is compared (G1-MST − B4). The algorithm is given in Section 3.7.
- **H2**: root mean square error of the global slope (mean of 59 points), G1-MST versus B4, with the same algorithm and decision rule as H1 (replacing "10 clusters" with a single global value).

**Rationale for placing H2 last (expected to be indistinguishable)**: noise in the global slope arises mainly from shifts of the entire examination — the SD of the examination-level global shift is about 1.45 dB, whereas it would be only 0.44 dB if the 59 points' noise were independent; borrowing information from neighbouring points within the same examination cannot remove a shared shift, and the shrinkage of linear mixed models is already close to optimal. In addition, after removing the global shift, the residual correlation between neighbouring points in the same hemifield at distance ≤ 6° reaches 0.507 (Section 6, item 3), indicating that the pointwise noise itself is spatially correlated (possibly related to the Octopus TOP strategy, which uses neighbouring responses to estimate thresholds; this is only a conjecture), which further reduces the scope for spatial borrowing. The value quoted in Section 0 of the plan, "mean inter-point correlation −0.017 after removing the shift", is an algebraic consequence of row-centring (approximately −1/58) and is not used as evidence for anything.

### 1.3 Scale of the ground truth

- **Primary analysis (observed scale)**: each point's true trajectory is first floored at 0 dB, i.e. \( \max(f(t), 0) \), and then regressed by OLS on the examination times within the observation window to obtain the pointwise true slope; cluster-level and global true values are the means of the corresponding pointwise true values. H1–H4 and the slope endpoints of Section 1.4 all use this scale.
- **Descriptive analysis (latent scale)**: the OLS slope of the unfloored true trajectory \( f(t) \). Reported only in supplementary tables.
- Rationale: after Revision R1 the template baseline may be below 0 dB; a point that stays below 0 dB is always observed as 0, and its latent slope cannot be recovered by any method; using the observed scale for the primary endpoint compares the ability of methods to recover slopes from observable information.

### 1.4 Other secondary and exploratory endpoints (descriptive only, no formal testing)

- For each method, scenario, and number of examinations (k = 3, 4, 5, 6): bias, root mean square error, and 95% interval coverage (raw and calibrated standard errors) of pointwise, cluster-level, and global slopes. Stratified by true rate, with bias reported specifically for fast-progressing eyes.
- The same set of metrics under the latent-scale ground truth (Section 1.3).
- Real data: detection proportions and Kaplan–Meier curves after calibration to a 5% false-positive rate by visit shuffling (Section 3.8).
- Real data: pointwise MAE of next-visit visual field prediction (history lengths 2 and 3), stratified by baseline MS tertile and by tertile of the full-series OLS slope, always listing "last value carried forward" (B0).
- The 50 eyes with at least 6 examinations: odd–even split-half slope reliability (intraclass correlation coefficient).
- Structure–function correlation between the denoised baseline visual field and the four RNFL sectors (Garway-Heath mapping, Spearman correlation, raw versus denoised).
- Agreement with the official PLR2, PLR3, and MD labels (supplementary table).
- Results of EB-GK on all of the above endpoints (Section 2.2).

### 1.5 Sample size statement

No formal power calculation is performed. The size of `sim_test` (about 80 eyes × 8 scenarios × 20 replicates) is fixed before generation; the effective resampling unit for the primary interval is the 43 subjects of the hold-out set, so the precision of conclusions is limited by the template diversity of these 43 subjects, which will be reported transparently. The real-data endpoint H4 has only 88 eyes, and its interval is expected to be wide.

---

## 2. Methods and comparators

### 2.1 Competing methods

**First batch (must be completed):**

| ID | Method | Key points |
|---|---|---|
| B0 | Last value carried forward | Prediction endpoints only |
| B1 | Pointwise linear regression (PLR) | Only with ≥3 examinations |
| B2 | MS linear regression | Global |
| B3 | Linear mixed model (MS) | Subject- and eye-level random intercepts and slopes |
| B4 | Spatial two-level mixed model | Following Zhao 2026: eye-level plus cluster-level random intercepts and slopes, plus an examination-level random intercept; **with a spatial term** (Section 6, item 3) |
| EB | Empirical Bayes shrinkage of pointwise slopes | Pointwise OLS slopes shrunk toward the cluster mean, weight τ²/(τ²+SE²), τ² estimated from training eyes stratified by sensitivity |
| B5 | Spatial smoothing followed by PLR | Nearest-neighbour smoothing, simple implementation |
| MST-S | G1-MST without temporal attention | Equivalent to the single-examination masked autoencoder of Wu 2024 |
| G1-MST | Proposed model | Section 2.5 |
| D1 | PoPLR | Detection endpoints only |

**Second batch (if resources permit)**: B6 VAE denoising, B8 Kalman filter, B9 small GRU, B10 deterministic pointwise-token Transformer (point-estimate version of Bern 2026). Methods not completed will be stated with reasons in the paper. No second-batch method enters H1–H4.

### 2.2 Descriptive comparator EB-GK (Revision R2)

| ID | Method | Key points |
|---|---|---|
| EB-GK | Gwak–Kang-style empirical Bayes shrinkage of pointwise slopes (descriptive comparator) | For each eye and point, the full-series OLS slope \( b \) and its standard error \( s \); shrinkage toward the **population mean slope of the training eyes at that location** \( \mu_p \); prior variance estimated per location \( \tau_p^2 = \max(0,\ \mathrm{var}_{train}(b_p) - \overline{s_p^2}) \); \( b_{GK} = \mu_p + w(b - \mu_p) \), \( w = \tau_p^2/(\tau_p^2 + s^2) \); parameters estimated on training eyes only. Cluster-level and global slopes are obtained by averaging the shrunk pointwise slopes as in Section 2.3 |

- Source: Gwak & Kang, MICCAI 2026 OMIA (OMIA_015). The original work shrinks only follow-up slopes and keeps the first visual field as an "anchor", in order to provide labels for a fundus-photograph model; the present study uses full-series slopes so that it can be compared with other methods on the same slope endpoints.
- Difference from EB: EB shrinks toward the same-cluster mean of the eye with τ² estimated in three sensitivity strata; EB-GK shrinks toward the training-population mean at that location with τ² estimated per location.
- **EB-GK is a descriptive comparator only**: it does not enter the H1–H4 fixed-sequence tests and does not take part in the decision point of Section 4.2.

### 2.3 Common interface and slope definitions

Each method is implemented as a class that takes a (truncated) series and outputs:

```python
class Method:
    def fit(self, train_series: list[Series]) -> None: ...
    def estimate(self, series: Series) -> Estimate: ...
    # Estimate fields:
    #   denoised: (n_visits, 59) denoised values (raw values for methods without denoising)
    #   slope_point, se_point: (59,)
    #   slope_cluster, se_cluster: (10,)
    #   slope_global, se_global: float
    #   forecast(t_new) -> (59,) predicted values, optionally with sigma
```

`Series` fields: `eye_id, subject, t (n,), y (n, 59), censored (n, 59) bool, age (n,), iop (n,)`. Censored observations are recorded as 0 dB with `censored = True`.

- **Cluster-level slope** (identical for all methods): the denoised values of each examination are first averaged within each cluster, then regressed on time by OLS. The central point (0,0) belongs to no cluster.
- **Global slope**: average over the 59 points, then OLS on time.
- Methods that directly produce cluster-level or pointwise slopes, such as B4, EB, and EB-GK, use their own estimates, which will be stated in the supplementary tables.
- When estimating slopes or making decisions, every method receives only the series truncated at the current examination in `estimate()`.
- **Estimation failure**: if a method fails to estimate a series (non-convergence, numerical overflow, etc.), the B1 estimate for that series is used in that method's results, and the failure rate is reported per method × scenario.

### 2.4 Implementation of comparator methods

| Method | Key implementation details |
|---|---|
| B0, B1, B2 | B1: pointwise OLS, standard error from the residuals of that point; B2: OLS on MS |
| B3 | statsmodels `mixedlm`: grouped by subject, random intercept and slope within subject, eye nested. For a new eye, variance components are fixed at the training-set values and only the best linear unbiased prediction (BLUP) for that eye is computed |
| B4 | Fixed effects: intercept + time. Random effects: eye (intercept, slope), cluster within eye (intercept, slope), examination (intercept), plus a spatial correlation term on the pointwise residuals. Procedure: estimate variance components on training eyes → for a new eye, fix the variance components and compute the BLUP, yielding cluster-level slopes and their standard errors |
| EB | Pointwise OLS slope \( b \) and standard error \( s \); in three sensitivity strata, prior variance estimated from training eyes \( \tau^2 = \max(0,\ \mathrm{var}(b) - \overline{s^2}) \); shrinkage toward the mean slope of the same cluster in that eye: \( b_{EB} = w b + (1-w)\bar b_{cluster} \), \( w = \tau^2/(\tau^2 + s^2) \) |
| B5 | Within each examination, each point is replaced by the weighted average of itself and its same-hemifield neighbours within ≤6° (self weight 0.5, neighbours sharing 0.5 equally), followed by PLR |
| D1 PoPLR | Pointwise OLS one-sided p values; statistic \( S = -\sum \ln p \) (over points with negative slopes only); permutation distribution from 1000 random shuffles of visit order. If R is available, 10 eyes are cross-checked against `visualFields` |

**Implementation of the B4 spatial term and fallbacks (pre-specified rules)**:
1. Preferred: Matérn spatial correlation in R `spaMM` (within hemifield only; no correlation across the horizontal midline).
2. In week 3 (2026-11-02 to 11-08), run time is measured on 20 randomly selected development-set eyes and extrapolated linearly in the number of eyes to the variance-component estimation on all development-set training eyes; if the projected time exceeds 24 hours, a conditional autoregressive structure on the neighbour graph (same hemifield, distance ≤ 6°) is used instead.
3. If variance-component estimation does not converge, then in order: ① the random-effects covariance is made diagonal; ② if still not converged, the conditional autoregressive structure is used.
4. The option finally adopted is appended as a timestamped revision before any method results are produced, and not changed thereafter.

### 2.5 G1-MST

**Architecture (about 30,000 parameters, fixed):**

- One token per point value. Embedding = linear layer(`[y/35, censored]`) + Fourier features of the coordinates (x, y) (8 frequencies) through a linear layer + Fourier features of time (years) through a linear layer + visit age as a scalar through a linear layer. Dimension d = 32.
- Each examination gets an additional "examination token" (a learnable initial vector plus time encoding) representing that examination's global shift.
- 2 layers of factorised attention, each with:
  - Spatial attention: the 59 point tokens plus 1 examination token within the same examination. Attention biases: +b (b learnable) for point pairs in the same hemifield at distance ≤ 6°, −∞ (not connected) for point pairs across the horizontal midline. The examination token is connected to all points.
  - Temporal attention: the same point across examinations; examination tokens also attend to each other over time.
  - Feed-forward width 64, LayerNorm, dropout 0.1.
- Output head: each point token outputs \( \mu \) and \( \log\sigma \).

**Training objective:**
- In each batch, one masking mode is chosen at random (equal probability) for each series: ① randomly mask 20% of point tokens; ② mask an entire examination (all 59 points); ③ mask all examinations of one point.
- Masked tokens have their value and censoring flag set to zero and receive a "masked" flag.
- The loss is computed on masked tokens only, using the censored Gaussian negative log-likelihood: uncensored \( -\log \mathcal N(y;\mu,\sigma) \); censored \( -\log \Phi\big((0-\mu)/\sigma\big) \).
- Optimisation: AdamW, learning rate 1e-3, weight decay 1e-2, batch size 16 series, at most 400 epochs; early stopping on the masked-token negative log-likelihood in the validation folds of the 5-fold split within the development set (patience 30 epochs).
- Training data: real data only (training folds of the development set). **The main model is not trained on simulated data.** A separate "real + `sim_train`" version (G1-MST-sim) is trained for sensitivity analysis SA7 only.
- 5 seeds: 20261019–20261023.

**Inference (obtaining denoised values):**
- For a (truncated) series, K = 20 inference passes are made, each randomly masking 20% of point tokens, with every token masked at least twice. The denoised value of each token = the mean of \( \mu \) over the passes in which it was masked. Thus no denoised value uses that point's own measurement at that examination (J-invariance principle, Batson & Royer 2019).
- Forecasting: a fully masked examination is appended at the target time and \( \mu, \sigma \) are read out (used only for next-visit prediction in Section 1.4, not for H4).

**Ablations** (in the first batch only MST-S; the others after the main results, all descriptive): removing temporal attention (MST-S), removing the examination token, removing the censored likelihood (MSE instead, −1 treated as 0), removing heteroscedastic σ, random point masking only, adding follow-up IOP, a 200,000-parameter version (d = 64, 3 layers).

### 2.6 Calibration of slope standard errors

Denoised values borrow information from each other, so standard errors computed directly with the OLS formula are too narrow. Procedure:
1. On `sim_train`, for each method and each level (pointwise, cluster, global), compute \( z = (\hat b - b_{true})/\widehat{SE} \) (\( b_{true} \) on the observed scale).
2. Calibration factor \( c = \) the 95% quantile of \( |z| \) / 1.96.
3. On `sim_test` and real data, use \( c \cdot \widehat{SE} \) as the calibrated standard error.
4. Both raw and calibrated coverage are reported.

All methods are calibrated this way (including B1 and B4) to ensure a fair detection comparison.

---

## 3. Data, noise model, simulator, and evaluation procedures

### 3.1 Data encoding

- Indices 21 and 32 of the 61 GRAPE values (physiological blind-spot placeholders) are removed, leaving 59 points.
- All other −1 values are left-censored observations at 0 dB: `censored = True`, value recorded as 0 dB.
- Locations, clusters, and the split were fixed before publication; see Section 6.

### 3.2 Population split rules

- The noise model, simulator parameters, tuning of all methods, and `sim_train` use development-set subjects only.
- `sim_test` uses hold-out subjects only as templates.
- G1-MST and other methods requiring training that are used in the simulation evaluation are trained only on real development-set data.
- Real-data endpoints (H4 and the real-data analyses of Section 1.4) use a 5-fold subject-level split of all 144 subjects (`all_folds`); in each fold every method is retrained on the other 4 folds.

### 3.3 Noise model (development set only)

For each eye and each point, OLS gives fitted values \( \hat f \) and residuals \( r \); residuals are multiplied by \( \sqrt{n/(n-2)} \).

1. **Examination-level global shift**: for each examination, \( \delta_v = \) the mean of the residuals of its 59 points; stratified by tertile of the eye's baseline MS (the three strata differ by more than 20%; see Section 6, item 4). For simulation, the deconvolved SD — after removing the contribution of the average pointwise noise — is used.
2. **Point-specific noise**: \( e = r - \delta_v \). The SD is estimated in bins of fitted value (0–5, 5–10, 10–15, 15–20, 20–25, 25–30, ≥30 dB) and then smoothed with a smoothing spline SD(\( \hat f \)).
3. **Spatial correlation**: pointwise noise is generated with an exponentially decaying spatial covariance within each hemifield, with the correlation length estimated from the data; no correlation across the horizontal midline. For noise generation the central point is assigned to the superior-hemifield block.
4. **Censoring**: simulated values < 0 are recorded as 0 with `censored = True`.

### 3.4 Simulator

Each simulated series = true trajectory + noise.

**Template (Revision R1)**: for each template eye, the **intercept of a pointwise Tobit regression (left-censored at 0 dB)** is used as the latent baseline at the time of the first examination (t = 0); it may be below 0 dB and is restricted to −10–35 dB.
- For each point of each template eye, \( y^* = a + b\,t + \varepsilon \) is fitted, with observations \( \max(y^*, 0) \) and −1 treated as \( y^* \le 0 \); \( \sigma \) is not estimated per point but fixed at \( \sqrt{SD_{point}(\hat f_{OLS})^2 + SD_{offset}^2} \) from the noise model (with the shift SD taken from the baseline-MS stratum); \( a, b \) are estimated by maximum likelihood. Without censoring at that point this equals OLS; if all examinations are censored the MLE diverges and the lower bound −10 dB is used.
- Rationale: with OLS intercepts (floored at 0 dB) as templates, the noise model's self-check simulation gave a censoring fraction of 2.7% versus 3.99% in the real development set, outside ±1 percentage point; 32% of censored points are absolute scotomata censored at every examination of that eye, which OLS fits at about 0 dB, so only about half of them fall below 0 in simulation. A diagnostic simulation that only set such points to −10 dB raised the fraction to 3.4%, indicating that the 0 dB floor is the main cause.

**Schedule**: the primary evaluation requires "exactly 4 examinations":
- If a template eye has ≥4 examinations, its first 4 time points are used; if it has fewer than 4, additional intervals are drawn with replacement from the empirical distribution of real consecutive intervals in the development set up to 4 examinations (`sim_test` also draws from the development-set interval distribution).
- Versions with k = 3, 5, and 6 examinations are also generated (with intervals drawn in the same way) for the "number of examinations versus error" curves.

**True-slope scenarios (8):**

| ID | Name | True trajectory | Positive/negative | "Favours B4"? |
|---|---|---|---|---|
| S0 | Stable with age-related decline | All points −0.1 dB/year | Negative | Yes |
| S1 | Cataract-like | All points drop by 1–2 dB once after a random examination, stable thereafter | Negative | No |
| S2 | Focal progression within clusters | All points in 1–2 random clusters at −0.5 / −1 / −2 dB/year (three levels with equal probability), others as S0 | Positive | Yes |
| S3 | Arcuate defect across clusters | An arc along the nerve-fibre course (spanning 2–3 clusters but covering only part of each cluster's points) at −1 / −2 dB/year | Positive | No |
| S4 | Small paracentral scotoma | 3–5 adjacent points within the central 10° at −1.5 / −3 dB/year | Positive | No |
| S5 | Diffuse progression | All points −0.3 / −0.6 dB/year | Positive | Yes |
| S6 | Late acceleration | Same locations as S2, −0.3 dB/year in the first half and −1.5 dB/year in the second half (piecewise linear, knot at the temporal midpoint) | Positive | No |
| S7 | Quadratic acceleration | Same locations as S3, trajectory \( -a t^2 \), with \( a \) such that the mean slope at the end of 4 examinations is −1 dB/year | Positive | No |

**Supplementary scenario rules**:
- "1–2", "3–5", and "2–3" in the table are integers drawn with equal probability; discrete levels such as "−1 / −2" are drawn with equal probability; the drop in S1 is drawn uniformly from [1, 2] dB.
- S1: the drop occurs before the \(j\)-th examination, with \(j\) drawn with equal probability from \(\{2, \dots, k\}\); the slope is 0 both before and after the drop (the age-related decline of S0 is not added). Under Section 1.3 the observed-scale true slope of S1 is generally negative, but S1 is counted as **negative** in the detection endpoints (used to set the specificity threshold); its true value is used as usual in the slope-error endpoints.
- In S2, S5, and S6, unaffected points are as in S0 (−0.1 dB/year).
- The point sets of the S3 arc and the S4 scotoma are generated automatically by the simulator code from the coordinates and the 10-cluster definition. After the simulator code is finalised and before any simulated data are generated, the point-set generation rules and the code's SHA-256 are appended as a timestamped revision.

**Size and seeds**:
- `sim_train`: each development-set template eye × 8 scenarios × 10 replicates, seed 20261019.
- `sim_test`: each hold-out template eye × 8 scenarios × 20 replicates, seed 20261119.
- Each series carries `scenario, template_eye, template_subject, truth_point, truth_cluster, truth_global (one set each on the observed and latent scales), is_progressing`.

**Sealing of `sim_test`**: immediately after generation, the file's SHA-256 is appended to the preregistration as a timestamped revision; **before 2027-01-04 (the first day of week 12)** no script may read it (the `sim_test_seal` switch in `config.yaml`; evaluation scripts check the switch and the hash). Weeks are counted with the week containing 2026-10-19 (Monday) as week 1.

### 3.5 Simulator calibration and the week-5 checkpoint (development-set real data and `sim_train` only)

| Statistic | Tolerance |
|---|---|
| MS residual SD | ±10% |
| Examination-level global shift SD | ±10% |
| Pointwise residual SD (per stratum of fitted value) | ±15% |
| SD of observed MS slopes | ±15% |
| Censoring fraction (target: 3.99% observed in the development set) | ±1 percentage point |

- **Mixing rule for the SD of observed MS slopes**: development-set eyes are classified by full-series OLS MS slope into three bins: ≥ −0.3 dB/year, −1.0 to −0.3 dB/year, < −1.0 dB/year; the proportions of eyes in the three bins are used as mixing weights for S0, S5, and S2, respectively; the S0, S5, and S2 series of `sim_train` are weighted accordingly, and the SD of observed MS slopes is computed and compared with the real development-set value.
- If a tolerance is exceeded, noise-model and simulator parameters are adjusted, for at most 3 rounds; if still out of tolerance, the gap is reported in the paper. Tuning looks only at the statistics in the table above and does not run any competing method.
- **Week-5 checkpoint** (2026-11-16 to 11-22): on `sim_train`, the cluster-level slope RMSE of B4 should be clearly lower than that of B1 (bootstrap over development-set subjects, interval excluding 0). If not, the simulator or the B4 implementation is fixed first, and the work does not proceed.

### 3.6 Detection on simulated data (H3)

1. Decision statistic of each method: the minimum over clusters of the cluster-slope z values (with calibrated standard errors); PoPLR uses S.
2. On the negative scenarios of `sim_train` (S0, S1), the threshold at which 95% of negative series are not called positive is determined separately for each method × seed × k.
3. The thresholds are applied to `sim_test`: the actual specificity of negative scenarios and the sensitivity of positive scenarios are reported, stratified by scenario and rate.
4. H3: the sensitivity difference at k = 4 for the focal progression scenarios (S2, S3, S4, S6, S7 combined, scenarios weighted equally) (G1-MST − B4; for G1-MST, the mean of the sensitivities of the 5 seeds). The 95% interval is obtained by resampling hold-out subjects with replacement 2000 times, with thresholds held fixed (not re-estimated) during resampling.
5. Additionally reported: "number of examinations needed to reach 80% sensitivity" (curve over k from 3 to 6).

### 3.7 Real-data anchor (H4)

1. Subjects: eyes with at least 5 examinations (88 eyes); 5-fold subject-level split of all 144 subjects, with each fold using models trained on the other 4 folds.
2. Each method calls `estimate()` on the first 4 examinations to obtain the intercept and slope of each cluster: methods in general fit OLS to the denoised cluster means of the 4 examinations; B4, EB, and EB-GK use their own cluster-level estimates. These are extrapolated to the times of the 5th and later examinations. G1-MST does not use its forecasting output but extrapolates cluster-level trends in the same way as other methods; for G1-MST the mean of the extrapolated values over the 5 seeds is used as the prediction.
3. Observed cluster mean: the mean of the observed values (censored recorded as 0 dB) of the points of that cluster at that examination.
4. MAE: for each eye, \( |\text{predicted} - \text{observed}| \) is first averaged over its 5th and later examinations × 10 clusters, and then averaged over eyes with equal weights.
5. Interval: 95% interval of the MAE difference (G1-MST − B4) from 2000 resamples of subjects with replacement.

### 3.8 Detection on real data (descriptive only)

1. For eyes with at least 5 examinations, the visit time labels of each eye are shuffled 200 times (time points unchanged, only the order of the values is permuted), and the decision statistic of each method is computed.
2. Threshold: the positive rate on shuffled series equals 5%.
3. On the true order, decisions are made sequentially from the 3rd examination onward, recording the examination number and years at first detection; Kaplan–Meier curves are drawn, with a subject-clustered log-rank test.
4. Limitation: shuffling removes age-related decline and cannot separate declines caused by cataract from those caused by treatment.

---

## 4. Sensitivity analyses and the narrative decision point

### 4.1 Sensitivity analyses (all descriptive; they do not change the H1–H4 decisions)

| ID | Change |
|---|---|
| SA1 | −1 treated as missing (v1 encoding), to assess the effect of censoring handling |
| SA2 | Excluding angle-closure eyes |
| SA3 | Generating `sim_test` only from template eyes with ≥4 examinations |
| SA4 | Excluding eyes whose follow-up IOP fell by more than 30% from baseline (no medication records are available; this serves as a proxy for treatment intensification) |
| SA5 | Detection specificity changed to 90% and 97.5% |
| SA6 | 200,000-parameter version of G1-MST; masking fractions of 10% and 30% |
| SA7 | G1-MST-sim (trained with added simulated data) |
| SA8 | Simulated noise without the examination-level global shift (pointwise noise only), to see whether conclusions depend on this component |
| SA9 | If Octopus age norms are available, H1 repeated with TD values |
| SA10 | Simulated pointwise noise made spatially independent, to see whether conclusions depend on spatially correlated noise |

### 4.2 Week-11 narrative decision point (2026-12-28 to 2027-01-03)

Using only `sim_train` and the development set: the difference in cluster-level slope RMSE between G1-MST and B4, with a paired bootstrap over development-set subjects (same algorithm as H1).

- Upper limit of the interval < 0: the abstract and title emphasise "a spatiotemporal denoising model outperforms a strong statistical model for cluster-level slopes".
- Interval containing 0 or B4 better: the title and abstract are framed as a "benchmark", emphasising "with small samples, strong statistical models are sufficient; global shifts dominate global noise".

**In either case, no endpoint, testing order, or decision rule of Section 1 is changed**; only the narrative emphasis of the paper is decided. The decision and its date are recorded in the log.

---

## 5. Pre-specified hyperparameters (`config.yaml` should be identical to this block verbatim)

> The block below is reproduced verbatim from `config.yaml`; comments and some string values are in Chinese. Scenario names: S0 stable with age-related decline; S1 cataract-like; S2 focal progression within clusters; S3 arcuate defect across clusters; S4 small paracentral scotoma; S5 diffuse progression; S6 late acceleration; S7 quadratic acceleration.

```yaml
# 方向 A：GRAPE 视野去噪方法基准 —— 全局配置（预先定死，预注册内容的一部分）
# 依据：方向A_研究计划_最终版.md（2026-10-20）。修改任何超参数都必须在预注册里带时间戳记录。
project: GRAPE_directionA_VF_denoising_benchmark
plan_version: "最终版 2026-10-20"
created: "2026-10-20"
revisions: ["R1 2026-10-20 Tobit 模板基线", "R2 2026-10-20 EB-GK 描述性对照"]

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
  target_censor_frac: 0.0399     # 开发集实测删失比例（校准只用开发集）

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
  sha256: null                 # 由 04_simulator.py 生成后写入，并追加为预注册修订

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

## 6. Data decisions fixed before publication (data structure only; no competing method has been run)

1. **Population split** (`data/split.json`, seed 20261019): development set 101 subjects / 182 eyes, hold-out set 43 subjects / 81 eyes; strata = subject baseline MS tertile × whether the subject has an eye with ≥5 examinations. Baseline MS distribution KS p = 0.92; examinations per eye Mann–Whitney p = 0.96.
2. **G1 location mapping** (`data/g1_locations.csv`): coordinates from `locmaps$g1` of the R package visualFields 1.0.7; the 61 GRAPE values are arranged in a centre-outward spiral according to Figure 2b of the original paper, verified point by point against the first examination of the right eye of subject 8 (59/59), with independent verification against the left-eye example of Figure 2c (58/59). The blind-spot placeholders take the centroid of the grey ellipse in Figure 2b (15.0°, −1.4°). All three validation checks passed.
3. **B4 spatial term**: after removing the examination-level global shift, the mean residual correlation of neighbouring points in the same hemifield at ≤6° = **0.507** (other pairs −0.035), > 0.10 → **B4 includes a spatial term**. The implementation is fixed in week 3 according to the rules of Section 2.4.
4. **Noise model** (`data/noise_model.json`, development set only):
   - The examination-level global shift SD in the three baseline-MS strata is 1.91 / 1.26 / 1.14 dB (relative difference 68% ≥ 20%) → **stratified**; simulation uses the deconvolved shift SDs 1.53 / 0.91 / 0.89 dB. Self-check: simulating with these and re-estimating with the same algorithm gives a global shift SD of 1.42 (real development set 1.47).
   - Pointwise noise: smoothing-spline SD as a function of fitted sensitivity; exponentially decaying spatial correlation within hemifields (a = 1.00, λ = 8.2°), no correlation across the horizontal midline.
   - This file was generated under the pre-R1 template convention (floor 0 dB); R1 takes only the σ-related parameters from it, and the censoring fraction is re-checked with the Tobit template during the calibration of Section 3.5.
5. **Cluster definition**: the 10 clusters of the Octopus EyeSuite G program (Bürki & Monhart, European Ophthalmic Review 2007, Figure 1b) are adopted; the central point (0,0) belongs to no cluster. Cluster numbering is defined by this study:

| Cluster | Name | Points | Points included (x°, y°), right-eye visual field coordinates, +x temporal |
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
| `data/split.json` | `5087d7813cfd43399b61a7f76ade8684c63c44da3128b66155665845ccb467a9` |
| `data/g1_locations.csv` | `6bda6f9e3f7cab9a5254701bf7cc482c0d941b1cd25478a02e3d1a6d9bf788af` |
| `data/noise_model.json` | `d0e0383d92843a6ebafe11df591c58a7893193e779006377dc48b52810367270` |

---

## 7. Revision history

| ID | Date (Asia/Shanghai) | Content | Decided by | Before any model results? |
|---|---|---|---|---|
| — | 2026-10-20 approx. 12:11 | Initial draft | Drafted by the execution agent | Yes |
| R1 | 2026-10-20 12:39 | Simulation template baseline changed to the pointwise Tobit (left-censored at 0 dB) intercept, allowing < 0 dB, range −10–35 dB; calibration still applies | Investigator | Yes (no simulated data generated yet) |
| R2 | 2026-10-20 12:39 | Descriptive comparator EB-GK added; not part of the H1–H4 fixed-sequence tests | Investigator | Yes (no method run yet) |
| S1 | 2026-10-20 13:11 | The `grape_pilot` directory was sealed as a whole into a deterministic tar.gz (packing started 13:10:34; 29,844 files including `.venv`), SHA-256 `8f37381cbbf3e9b64601a61c684f0ac7a536cbd8a5510cfb01be2afb9c60feb0`, with a per-file SHA-256 list `grape_pilot_manifest.sha256`; during sealing only file names and sizes were listed and no numbers in it were viewed. Note: the latest file modification time in the directory is 13:09, later than the 12:35 stated in Section 0.1, item 4 | Requested by the investigator, executed by the execution agent | Not applicable (sealing only; no new results produced) |
| S2 | 2026-10-20 15:47 | `grape_pilot` resealed. The 13:11 seal (S1, `8f37381cbbf3e9b64601a61c684f0ac7a536cbd8a5510cfb01be2afb9c60feb0`) is superseded because the directory was written to after it; the old archive and manifest are kept, renamed `*_v1_1311`. New deterministic tar.gz packed 15:46:05–15:46:57: 29,857 files including `.venv` and `paper/`, SHA-256 `a8873d3e9335bcede283d3d879ef34a189edc7208dcc45dfee5b4ffaaa0cb24f`. Re-packing reproduced the hash, and the per-file manifest check after extraction passed. Again, only file names and sizes were listed and no numbers were viewed. Checked before sealing: no running processes in the directory and no writes after 15:34:45. Timeline confirmed by view1, the agent that ran the pilot (Beijing time, 2026-10-20): started approx. 11:50; interrupted and resumed approx. 12:47–12:53; computation ended 13:33 (last computational output `out/e_summary.txt` 13:31, `phase2_report.md` 13:33); afterwards only two summary documents were written (15:25 `总结_符合与偏离预期.md` and 15:34 `view1_试跑总报告.md`); last write 15:34:45. Copies outside the directory (six `/workspace/view1数据*.zip` files and `/workspace/out/方向A_动笔前修改清单.md`) are recorded by SHA-256, unopened, in `sealed/external_copies.sha256` (SHA-256 of that file: `c8395d351d31fce2c6cc00621e0bd8d9e143339f069d8f0c58505a6cc32db69a`) | Approved by the investigator, executed by the execution agent | Not applicable (sealing only; no new results produced) |
| S3 | 2026-10-20 16:05 | At the investigator's request, Section 0.1, item 4 was rewritten using known metadata only (file names, modification times, hashes, and view1's own account of the timeline), without opening any file in the pilot directory or its external copies. Added: the actual timing and scope of the pilot; the v2 delivery time of 13:05:22 (attachment modification time); the existence of pilot output files from 11:51 onward; and the execution agent's single opening of `pilot_report.md` | Investigator | Not applicable (disclosure only; no new results produced) |
Any change after publication is appended as a new row (with date, content, and rationale) and reported item by item in the paper. Announced forthcoming revisions: implementation of the B4 spatial term (week 3); point-set generation rules for S3/S4 and the simulator code hash (before generating simulated data); the SHA-256 of `sim_test` (immediately after generation).
