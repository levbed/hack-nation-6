# mcPHASES CycleBench Data Card

## Sources

### mcPHASES

CycleBench uses mcPHASES v1.0.0:

> Lin, B., Li, J. Y., Kalani, K., Truong, K., & Mariakakis, A. (2025).
> mcPHASES: A Dataset of Physiological, Hormonal, and Self-reported Events and
> Symptoms for Menstrual Health Tracking with Wearables. PhysioNet.
> https://doi.org/10.13026/zx6a-2c81

The source dataset contains longitudinal data from 42 Canadian young adult
participants across two study intervals. Modalities include Fitbit data,
continuous glucose monitoring, urine hormone measurements, menstrual reports,
and daily self-reported experiences.

Official dataset page: https://physionet.org/content/mcphases/1.0.0/

### Utah cycle-length cohort

The independent history-only replication uses:

> Najmabadi, S., & Stanford, J. (2023). Menstrual Cycles Length of Women in
> the USA and Canada, 1990-2013. The Hive: University of Utah Research Data
> Repository. https://doi.org/10.7278/S50d-4gxs-s4hj

The file contains start and end dates for 3,324 cycles from 581 participants
across three studies conducted from 1990 to 2013. Its columns are participant
ID, age, cycle number, start date, end date, cycle length, and conception-cycle
status. CycleBench uses only participant ID, cycle number, dates, and length.

## Access And License

mcPHASES is a restricted-access PhysioNet resource governed by the PhysioNet
Restricted Health Data License and Data Use Agreement 1.5.0. Each researcher
must obtain access directly from PhysioNet and comply with its terms.

This repository does not redistribute mcPHASES files, participant records,
participant identifiers, or participant-level derived outputs. The repository's
MIT license applies to CycleBench code and documentation, not to mcPHASES data.

The Utah record is public under CC BY-NC. `download-utah` retrieves its CSV and
README directly from the official Hive record and validates pinned checksums.
Users remain responsible for attribution and license compliance.

Both raw datasets are stored only in ignored local directories. This repository
does not redistribute either dataset or participant-level derived outputs.

## Tables Used

CycleBench automatically uses available variables from:

| Source table | Variables |
| --- | --- |
| `hormones_and_selfreport.csv` | Menstrual evidence, E3G/estrogen, LH, PdG, FSH when present, and ordinal self-reports |
| `sleep.csv` | Minutes asleep, minutes awake, time in bed, sleep efficiency |
| `resting_heart_rate.csv` | Resting heart rate |
| `active_minutes.csv` | Light, moderate, and vigorous active minutes |
| `steps.csv` | Daily total steps |
| `glucose.csv` | CGM glucose values |
| `stress_score.csv` | Fitbit stress score |

The Utah replication reads only `CrMcyclelength_share.csv`; it has no hormone,
wearable, symptom, glucose, or stress feature track.

Self-reports include appetite, exercise level, headaches, cramps, sore breasts,
fatigue, sleep issues, mood swings, stress, food cravings, indigestion, and
bloating. Ordered responses are mapped from 0 (`Not at all`) through 5
(`Very High`). Unknown values are treated as missing.

## Unit Of Analysis

One example represents a complete source cycle `t` and its immediately
following complete target cycle `t+1`, within one participant and study
interval. The label is the target cycle's length in days.

Features use only source-cycle measurements with
`source_start_day <= day < target_start_day`. The target-cycle end is used only
to calculate the label and is never included in model features.

For Utah, supplied dates are interpreted as inclusive. Cycle length is checked
against `end_date - start_date + 1`. Sequential cycle numbers and adjacent dates
are required, and target end is retained only as label provenance.

## Eligibility

- At least three consecutive inferred cycle starts are required.
- Source and target cycle lengths must each be between 10 and 90 days inclusive.
- Menstrual episodes are inferred from positive flow or `Menstrual` phase
  reports; evidence days separated by at most two days are merged.
- Missing modalities do not exclude an otherwise eligible cycle pair.

For Utah:

- Source and target rows must have recorded lengths between 10 and 90 days.
- Cycle numbers must increase by one and the target must begin one day after
  the source ends.
- Missing, out-of-range, or non-adjacent cycles reset history; gaps are never
  bridged.
- Age and conception-cycle status are not used as features.

## Known Limitations

- Cycle boundaries are inferred from self-report rather than adjudicated.
- The cohort is small and does not represent all ages, geographies, health
  conditions, devices, or menstrual experiences.
- Hormone and sensor coverage differs across participants and cycles.
- Ordinal self-report mappings assume consistent interpretation of response
  categories.
- CycleBench does not establish clinical validity or causal relationships.
- The Utah source selected participants with regular menstrual bleeding and
  does not provide the modalities needed to replicate multimodal tracks.
- Differences in collection period, population, cycle adjudication, and
  eligibility prevent direct comparison of absolute error across cohorts.

## Privacy

Raw data and generated participant-level files are ignored by Git. OpenAI
summarization accepts only `benchmark_summary.json`, which contains aggregate
cohort counts, coverage, model metrics, and uncertainty intervals. It rejects
participant IDs, example IDs, dates, and predictions.

The static results explorer uses the same validated aggregate summary. It does
not contain participant-level rows or make network requests to a model API.
