# mcPHASES CycleBench Benchmark Card

## Research Question

For participants with complete consecutive menstrual cycles, do source-cycle
hormone, wearable, symptom, glucose, or stress summaries improve prediction of
the next cycle's length beyond menstrual-history baselines? Does the
history-only model ranking replicate in a larger independent cohort?

## Task

For source cycle `t`, predict the length in days of cycle `t+1`. Cycle `t+1`
begins at the next inferred menstrual start and ends at the following inferred
menstrual start.

In mcPHASES, starts are inferred from menstrual evidence. In Utah, complete
cycle start/end dates are supplied and validated. The target definition and
10-to-90-day range are otherwise shared.

This is an exploratory research task. It is not intended for diagnosis,
fertility planning, treatment, individual guidance, or perimenopause prediction.

## Tracks

1. `global_median`: target median fitted on the outer training fold.
2. `previous_cycle`: the complete source-cycle length.
3. `history_only`: previous length, historical mean, median, standard deviation,
   and prior-cycle count.
4. `history_plus_wearables`: history plus sleep, resting heart rate, activity,
   steps, and coverage summaries.
5. `history_plus_hormones`: history plus E3G, LH, PdG, FSH, and coverage
   summaries when available.
6. `history_plus_symptoms`: history plus ordinal self-report summaries.
7. `history_plus_glucose_stress`: history plus CGM and Fitbit stress summaries.
8. `full_multimodal`: all available feature families.

Unavailable variables and unavailable optional tracks are skipped and reported.
Utah therefore evaluates only the two baselines and `history_only`.

## Models And Preprocessing

- Training-fold median baseline.
- Previous-cycle baseline.
- Ridge regression as the primary linear analysis.
- RBF support vector regression and `HistGradientBoostingRegressor` as
  nonlinear sensitivity analyses.
- Training-fold median imputation for every trained model and standard scaling
  for Ridge and RBF-SVR.
- Model-specific fixed hyperparameter grids selected by MAE using inner
  participant-disjoint GroupKFold.
- Boosting uses fixed iteration counts with automatic early stopping disabled.

Feature availability, imputation, scaling, parameter selection, and model fitting
occur without access to the outer test fold.

## Evaluation

- Outer participant-disjoint `GroupKFold`, grouped by participant ID.
- Five folds when at least five participants are available; otherwise three
  folds when feasible.
- MAE, RMSE, median absolute error, percentage within 3 days, percentage within
  7 days, and mean signed error.
- Paired MAE difference versus `history_only`.
- 95% participant-clustered bootstrap intervals using 2,000 deterministic
  replicates.

Evaluation is independent within each cohort. The benchmark does not pool
cohort rows or preprocessing. Cross-cohort tables preserve cohort labels;
unweighted macro scores include only model/track pairs present in every cohort.

## Leakage Controls

- Training and test participant sets must be disjoint in every outer fold.
- Feature timestamps must be strictly before the target cycle begins.
- No target-cycle measurement, target end date, or future-cycle summary is a
  model feature.
- All learned preprocessing and hyperparameter selection is contained within
  training data.
- Utah histories reset across non-adjacent dates, skipped cycle numbers,
  missing lengths, and out-of-range cycles.

Automated tests enforce participant separation, temporal boundaries, aggregate
OpenAI payload safety, and the use of `store=False` for OpenAI requests.

## v2.1 Aggregate Results

Local evaluation of mcPHASES v1.0.0 produced 82 examples from 42 participants
and 142 inferred complete cycles.

| Model | History | + Hormones | + Wearables | + Symptoms | + Glucose/stress | Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 4.84 | 5.17 | 5.39 | 5.28 | 5.23 | 5.83 |
| RBF-SVR | **4.54** | 4.95 | 4.81 | 4.90 | 4.80 | 4.92 |
| HistGradientBoosting | 4.73 | 5.10 | 5.24 | 5.27 | 5.01 | 5.38 |

No added modality improved over history-only within its model family. RBF-SVR
history-only had the lowest point estimate, but its paired MAE difference from
Ridge history-only was inconclusive. These are benchmark comparisons, not
causal or clinical findings.

The Utah replication produced 2,408 examples from 581 participants and 3,144
cycles with recorded lengths:

| Model | History MAE | 95% participant-bootstrap CI |
| --- | ---: | ---: |
| RBF-SVR | **2.87** | 2.63-3.13 |
| HistGradientBoosting | 2.94 | 2.71-3.18 |
| Ridge | 2.97 | 2.74-3.21 |
| Previous cycle | 3.46 | 3.18-3.76 |
| Training-fold median | 3.55 | 3.24-3.88 |

RBF-SVR's MAE difference from Ridge was -0.091 days (95% paired interval
-0.161 to -0.023). This is a small improvement in one cohort, not evidence of
clinical relevance. Absolute mcPHASES and Utah errors should not be directly
compared because the cohorts and boundary definitions differ.

## Outputs

The pipeline writes aggregate scores, fold diagnostics, participant-level
predictions, aggregate reports, and plots. Generated outputs remain ignored by
default. `export-public` validates the aggregate summary before it is copied to
the static explorer. Only that artifact and aggregate figures should be
committed.

`evaluate-all` stores each cohort under a separate subdirectory and creates
aggregate-only cohort and macro tables. It does not create a pooled prediction
file.
