---
title: Maple Physio No-Show Prediction
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
python_version: "3.12"
---

# Maple Physio no-show check

## What it does

Maple Physio is a fictional physiotherapy clinic where missed appointments leave unused booking slots. This project estimates the chance that a patient will miss an appointment and recommends whether the front desk should call the patient the day before.

The assignment assumes a confirmation call costs **$8**, an uncalled no-show costs **$35**, and a called patient attends. The goal is to reduce the total cost compared with calling nobody or calling everybody.

The application uses Python, scikit-learn, pandas, joblib, and a Gradio Blocks interface.

**Public Hugging Face Space:** https://huggingface.co/spaces/amreleraqi09/midterm-no-shows

## How to run it locally

Clone the repository and enter the project folder:

```bash
git clone https://github.com/amr7772/midterm-no-shows.git
cd midterm-no-shows
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
```

Recreate the model and reports, if needed:

```bash
python -m noshows.train
```

Start the app:

```bash
python app.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:7860`).

## Team

| Member | Work owned |
|---|---|
| Amr Mohamed | Repository setup and coordination; Hugging Face Space hosting |
| Michael Yen | Preprocessing pipeline, model comparison, and tuning |
| Orcun Ercan | Cost function, threshold selection, training, and saved model/reports |
| Miora Gaelle Rakotoarivony | `Appointment` and `NoShowPredictor` classes and their tests |
| Sarthak Suman | Gradio app, class integration, input/error handling, Space settings, and deployment tests |

Each member's contribution is documented through GitHub commits and pull requests.

## Decisions

### Data and split

`noshows/data.py` generates 4,000 fictional appointments. The project uses a stratified 75/25 split (3,000 training and 1,000 test appointments), with random state 42. Model comparison, tuning, and threshold selection are performed without using the held-out test set.

### Inputs and validation

Only information that could be available when an appointment is booked is used as a feature. The `Appointment` class validates inputs and provides the one-row table expected by the trained model.

| Input feature | Accepted value | Reason for use |
|---|---|---|
| `lead_days` | 0–120 | Booking lead time may relate to attendance. |
| `age` | 12–110 | Attendance patterns may vary with age. |
| `prior_no_shows` | 0–30 | Prior missed visits may predict future no-shows. |
| `hour` | 8–17 | Appointment time may relate to attendance. |
| `weekday` | `mon`, `tue`, `wed`, `thu`, `fri` | Attendance may vary by weekday. |
| `booking` | `online`, `phone`, `front_desk` | Booking method may relate to attendance. |
| `insurance` | `private`, `provincial`, `self_pay` | Insurance category may relate to attendance. |
| `treatment` | `physio`, `massage`, `acupuncture` | Appointment type may relate to attendance. |
| `first_visit` | 0 or 1 | First-time and returning patients may differ. |
| `distance_km` | 0.1–150, or blank | Travel distance may affect attendance; blank is permitted. |

Unacceptable values trigger an error identifying the field. When distance is missing, the interface warns the receptionist; the numeric preprocessing pipeline uses median imputation.

### Excluded columns

| Column | Why it is excluded |
|---|---|
| `appointment_id` | A row identifier, not an appointment characteristic. |
| `no_show` | The target outcome to predict, not an input feature. |
| `fee_charged` | Recorded *after* a visit; using it would leak information unavailable at booking time. |

### Preprocessing and decision rule

The scikit-learn pipeline applies median imputation and scaling to numeric columns and one-hot encoding to categorical columns. The encoder tolerates unknown categories internally, but the `Appointment` class rejects unsupported choices rather than silently scoring them.

`NoShowPredictor` loads the fitted pipeline and `model_info.json`, then returns the estimated no-show probability and recommended action. The decision threshold is **read from `model_info.json`**, not written directly into the app. The saved threshold is **0.20**: a probability at or above 20% results in a recommendation to call.

## Model comparison

Four classifiers were compared with five-fold stratified cross-validation on the training portion only. The following values come from `reports/cv_results.csv` and are rounded to three decimal places.

### Mean cross-validation scores

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy classifier | 0.738 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression (balanced) | 0.669 | 0.415 | 0.639 | 0.503 | 0.713 |
| Random forest | 0.744 | 0.532 | 0.181 | 0.270 | 0.679 |
| Histogram gradient boosting | 0.730 | 0.470 | 0.248 | 0.325 | 0.667 |

### Standard deviations across folds

| Model | Accuracy SD | Precision SD | Recall SD | F1 SD | ROC-AUC SD |
|---|---:|---:|---:|---:|---:|
| Dummy classifier | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Logistic regression (balanced) | 0.016 | 0.017 | 0.027 | 0.014 | 0.022 |
| Random forest | 0.014 | 0.075 | 0.030 | 0.043 | 0.023 |
| Histogram gradient boosting | 0.005 | 0.021 | 0.020 | 0.021 | 0.015 |

Balanced logistic regression had the highest mean ROC-AUC, recall, and F1 among the candidates. Random forest had slightly higher accuracy but much lower recall. The dummy classifier reached 73.8% accuracy while detecting no no-shows, illustrating why accuracy alone is misleading here.

Logistic regression was selected for tuning. According to the team's reported tuning results, its ROC-AUC changed from approximately **0.7129** (untuned balanced candidate) to **0.7131** (best tuned result), a gain of only **0.0002**. This gain is much smaller than the approximately **0.022** cross-validation standard deviation, so the tuning improvement is not meaningful relative to the fold-to-fold spread.

The team's reported selected settings were `C=0.1`, `class_weight=None`, `max_iter=1000`, and `random_state=42`. The saved configuration in `models/model_info.json` is the source of truth for final settings. The tuned model is not the same as the initial balanced candidate because its selected `class_weight` is `None`.

## Result

The final model was evaluated once on **1,000 held-out appointments**. The figures below come from `reports/metrics.json`.

### Test-set performance

| Metric | Value |
|---|---:|
| Decision threshold | 0.20 |
| Precision | 0.381 |
| Recall | 0.821 |
| F1 | 0.520 |
| ROC-AUC | 0.761 |

| Actual outcome | Predicted no call | Predicted call |
|---|---:|---:|
| Attended | 388 | 350 |
| No-show | 47 | 215 |

The model found **215 of 262** actual no-shows but also flagged **350** patients who would have attended. Its recall is high, while precision is relatively low; the selected trade-off reflects the assignment's call-versus-missed-visit costs.

### Cost comparison

| Strategy | Test-set cost |
|---|---:|
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| Use the model | **$6,165** |

The model reduced simulated costs by **$3,005 (32.8%)** versus calling nobody and **$1,835 (22.9%)** versus calling everybody. On out-of-fold training predictions, the lowest reported cost was **$19,474** at the selected **0.20** threshold. The threshold was not selected using the test set.

These savings are estimates on one simulated test set of 1,000 appointments, not guaranteed clinic savings. With a different sample, patient population, or real-world cost assumptions, the cost advantage could change.

### Leakage check

| Feature set | Cross-validated ROC-AUC |
|---|---:|
| Honest features only | 0.713 |
| Including `fee_charged` | 0.923 |

The apparently large improvement from `fee_charged` cannot be trusted for deployment: that value is recorded after an appointment, so it would not exist when the receptionist makes a decision.

## Tests

The repository uses pytest for the data, split, pipeline, costs, threshold decisions, appointment validation, prediction behavior, model loading, and deployment-file consistency. Tests include parametrized input checks. The six deployment tests added with Part 5 check the saved model files, dependency versions, README Space settings, and `app.py` presence.

From the repository root, run:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

From a fresh clone of the final `main` branch, **61 tests passed**, and both Ruff checks passed.

## Limitations

- **Simulated data:** The dataset is synthetic and may not reflect real clinic patients or booking practices. Results should be validated on real, appropriately governed data before practical use.
- **Evaluation uncertainty:** Only one held-out test split of 1,000 appointments was used. Estimated costs and scores will vary across samples and settings.
- **Incomplete context:** Unexpected illness, emergencies, changes of plans, and transport disruptions are not represented directly.
- **Missing distance:** Median imputation lets the system run, but it cannot recover the patient's actual travel distance.
- **Cost assumptions:** The model assumes every call costs $8 and prevents a no-show, while an uncalled missed visit costs $35. Real outcomes and costs may differ.
- **Decision support only:** The prediction is a probability, not a certainty; reception staff should treat the call recommendation as an aid rather than an automatic clinical decision.


