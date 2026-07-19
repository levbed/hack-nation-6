# mcPHASES CycleBench

mcPHASES CycleBench is a participant-disjoint benchmark for next-cycle length
forecasting. It tests whether hormone, wearable, symptom, glucose, or stress
summaries improve prediction beyond simple menstrual-history baselines and
whether nonlinear models change that conclusion.

The project is an exploratory research benchmark, not a clinical application.
It is not intended for diagnosis, fertility planning, treatment, individual
guidance, or perimenopause prediction.

## Research Question

For participants with complete consecutive menstrual cycles, do measurements
from source cycle `t` improve prediction of the length of cycle `t+1` over cycle
history alone?

## Dataset Access

Download mcPHASES v1.0.0 through its official restricted-access PhysioNet page:

https://physionet.org/content/mcphases/1.0.0/

Each researcher must sign and follow the PhysioNet data-use agreement. This
repository does not redistribute raw data or participant-level derived outputs.
The MIT license applies to CycleBench code, not to mcPHASES data.

The loader accepts a directory containing `hormones_and_selfreport.csv`, or a
parent containing exactly one extracted release directory. It reads headers for
all CSV tables but scans rows only for tables used by the benchmark. See
[`DATA_CARD.md`](DATA_CARD.md) for tables, transformations, licensing, and
limitations. Cite the source dataset using DOI `10.13026/zx6a-2c81`.

## Target And Eligibility

For source cycle `t`, predict the length in days of cycle `t+1`.

Cycle starts are inferred from positive flow or `Menstrual` phase reports.
Menstrual-evidence days separated by at most two days are merged into one
episode. An example requires three consecutive starts in the same participant
and study interval. Source and target lengths must each be within 10 to 90 days.

Features use only measurements satisfying:

```text
source_start_day <= feature_day < target_start_day
```

The target-cycle end is used only to calculate the label. It is never a model
feature.

## Feature Tracks

- `global_median`: target median fitted on the outer training fold.
- `previous_cycle`: complete source-cycle length.
- `history_only`: previous length, historical mean, median, standard deviation,
  and prior-cycle count.
- `history_plus_wearables`: history plus sleep, resting heart rate, steps, and
  activity summaries.
- `history_plus_hormones`: history plus E3G, LH, PdG, and FSH when available.
- `history_plus_symptoms`: history plus ordinal daily self-reports.
- `history_plus_glucose_stress`: history plus CGM and Fitbit stress summaries.
- `full_multimodal`: all available feature families.

Each measured variable contributes its mean, maximum, standard deviation,
observation count, and source-cycle day coverage. Unavailable variables and
optional tracks are skipped automatically and reported.

## Models And Evaluation

The primary analysis is Ridge regression. RBF-SVR and
`HistGradientBoostingRegressor` are pre-specified nonlinear sensitivity
analyses. The two simple baselines are evaluated on the same outer folds.

Every trained model uses nested participant-disjoint cross-validation:

- Outer `GroupKFold` by participant, with five folds when feasible.
- Inner `GroupKFold` by participant, selecting hyperparameters by MAE.
- Training-fold median imputation for every model.
- Standard scaling inside the Ridge and RBF-SVR pipelines.
- Conservative fixed grids for Ridge regularization, RBF-SVR `C` and
  `epsilon`, and boosted-tree complexity and regularization.
- No automatic boosting early stopping or preprocessing outside training data.

The benchmark reports MAE, RMSE, median absolute error, percentage within 3 and
7 days, mean signed error, paired MAE differences, and 95% participant-clustered
bootstrap intervals. See [`BENCHMARK_CARD.md`](BENCHMARK_CARD.md) for the full
protocol.

## Installation

Python 3.10 or newer is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The `inspect` command uses only Python's standard library. Evaluation and
plotting require the installed dependencies.

## Reproduction

Replace `/path/to/mcphases` with the extracted local dataset directory:

```bash
python3 run_benchmark.py inspect --data-dir /path/to/mcphases

source .venv/bin/activate
python run_benchmark.py evaluate --data-dir /path/to/mcphases
python -m unittest discover -s tests -v
```

Global options such as `--output-dir` must appear before the subcommand.

## Aggregate Results

The verified mcPHASES v1.0.0 run found 42 participants, 142 inferred complete
cycles, and 82 eligible examples. MAE values are days:

| Model | History | + Hormones | + Wearables | + Symptoms | + Glucose/stress | Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 4.84 | 5.17 | 5.39 | 5.28 | 5.23 | 5.83 |
| RBF-SVR | **4.54** | 4.95 | 4.81 | 4.90 | 4.80 | 4.92 |
| HistGradientBoosting | 4.73 | 5.10 | 5.24 | 5.27 | 5.01 | 5.38 |

The global-median and previous-cycle baselines had MAE 5.29 and 5.68 days.
RBF-SVR history-only had the lowest point MAE, but its paired difference from
Ridge history-only was inconclusive: -0.301 days with a 95% participant
bootstrap interval of -0.975 to 0.358. No added modality improved over the
corresponding model's history-only track. RBF-SVR improved over Ridge on the
full-multimodal track, but RBF-SVR full multimodal did not improve over RBF-SVR
history-only.

![Model-by-track MAE](docs/figures/model_track_heatmap.png)

![Nonlinear comparison with Ridge](docs/figures/model_delta_vs_ridge.png)

These are benchmark comparisons, not causal or clinical findings.

## Results Explorer

The static explorer reads only a validated aggregate artifact. It contains no
participant identifiers, dates, predictions, raw measurements, or API keys.

```bash
python run_benchmark.py export-public \
  --results-dir results \
  --output-file docs/data/benchmark_summary.json
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. The explorer supports model and track selection,
six metrics, absolute and history-relative views, MAE intervals, and a
model-by-track matrix. The Pages workflow deploys `docs/` after GitHub Pages is
configured to use GitHub Actions.

## Outputs

Evaluation writes:

- `results/scores.csv`
- `results/fold_scores.csv`
- `results/predictions.csv`
- `results/benchmark_summary.json`
- `results/benchmark_report.md`
- `results/mae_by_track.png`
- `results/mae_delta_vs_history.png`
- `results/model_track_heatmap.png`
- `results/model_delta_vs_ridge.png`
- `results/predicted_vs_observed.png`
- `results/target_distribution.png`

Generated results are ignored by default. `predictions.csv` and the
predicted-versus-observed plot are participant-level derived outputs and must
not be committed. `export-public` validates and writes the aggregate artifact
used by the static explorer.

## Optional OpenAI Report

After evaluation, an optional command uses the OpenAI Responses API to explain
the aggregate benchmark summary:

```bash
export OPENAI_API_KEY="your-key"
python run_benchmark.py summarize --results-dir results
```

The default model is `gpt-5.6-terra`; override it with `--model`. The request
uses Structured Outputs, low reasoning effort, and `store=False`. It reads only
the validated aggregate summary and never uploads raw mcPHASES data or
participant-level predictions. The default output budget is 4,000 tokens; use
`--max-output-tokens` to override it.

## Limitations

Cycle boundaries are inferred rather than adjudicated. The cohort and eligible
example count are modest, modalities have unequal coverage, and the same
dataset supports both model development and evaluation design. The benchmark
uses source-cycle summaries rather than raw temporal trajectories and has no
external or prospective validation. Results should not be generalized beyond
this protocol.
